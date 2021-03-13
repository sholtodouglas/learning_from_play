import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import Progbar
from tensorflow.distribute import ReduceOp
from tensorflow.keras import mixed_precision
import tensorflow_probability as tfp
tfd = tfp.distributions
import lfp
import os
import wandb
import json

from lfp.metric import record, log_action_breakdown


class BetaScheduler():
    def __init__(self, schedule='constant', beta=0.0, beta_max=1.0, max_steps=1e4,
                 cycles=10, duty_cycle=0.5, plot=True):
        self.schedule = schedule
        self.beta_min = beta
        self.beta_max = beta_max
        self.max_steps = max_steps
        self.cycles = cycles
        self.duty_cycle = duty_cycle

        if schedule=='constant':
            self.scheduler = lambda s: tf.ones_like(s, dtype=tf.float32)*beta
        elif schedule=='linear':
            self.scheduler = self.linear_schedule
        elif schedule=='quadratic':
            self.scheduler = self.quadratic_schedule
        elif schedule=='cyclic':
            self.scheduler = self.cyclic_schedule
        else:
            raise NotImplementedError()
        if plot: self._plot_schedule()

    def linear_schedule(self, step):
        beta = self.beta_min + (step) * (self.beta_max-self.beta_min)/self.max_steps
        return tf.clip_by_value(float(beta), self.beta_min, self.beta_max, name='beta_linear')

    def quadratic_schedule(self, step):
        ''' y = (b1-b0)/n^2 * x^2 + b0 '''
        beta = self.beta_min + (step)**2 * (self.beta_max-self.beta_min)/self.max_steps**2
        return tf.clip_by_value(float(beta), self.beta_min, self.beta_max, name='beta_quadratic')

    def cyclic_schedule(self, step):
        period = self.max_steps // self.cycles
        step = step % period # map step to cycle
        if step < period * self.duty_cycle:
            # linear regime
            beta = self.beta_min + (step) * (self.beta_max-self.beta_min)/(period*self.duty_cycle)
        else:
            # constant regime
            beta = self.beta_max
        return tf.clip_by_value(float(beta), self.beta_min, self.beta_max, name='beta_cyclic')

    def _plot_schedule(self):
        ts = np.arange(self.max_steps, step=100)
        plt.plot(ts, [self.scheduler(t) for t in ts])
        plt.xlabel('Steps')
        plt.ylabel('Beta')




class LFPTrainer():

    def __init__(self, args, actor, dl, encoder=None, planner=None, cnn=None, optimizer=Adam(), strategy=None, global_batch_size=32):

        self.actor = actor
        self.encoder = encoder
        self.planner = planner
        self.cnn = cnn
        self.strategy = strategy
        self.args = args
        self.dl = dl
        self.global_batch_size = global_batch_size

        if args.fp16:
            optimizer = mixed_precision.LossScaleOptimizer(optimizer)

        if self.args.num_distribs is None: # different sized clips due to different sized losses
            actor_clip  = 0.06
            encoder_clip = 0.03
            planner_clip = 0.001
        else:
            actor_clip = 400
            encoder_clip = 5
            planner_clip = 0.4

        self.actor_optimizer = optimizer(learning_rate=args.learning_rate, clipnorm=actor_clip)
        self.encoder_optimizer = optimizer(learning_rate=args.learning_rate, clipnorm=encoder_clip)
        self.planner_optimizer = optimizer(learning_rate=args.learning_rate, clipnorm=planner_clip)

        self.nll_action_loss = lambda y, p_y: tf.reduce_sum(-p_y.log_prob(y), axis=2)
        self.mae_action_loss = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
        self.mse_action_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

        self.metrics = {}
        self.metrics['train_loss'] = tf.keras.metrics.Mean(name='train_loss')
        self.metrics['valid_loss'] = tf.keras.metrics.Mean(name='valid_loss')
        self.metrics['actor_grad_norm'] = tf.keras.metrics.Mean(name='actor_grad_norm')
        self.metrics['encoder_grad_norm'] = tf.keras.metrics.Mean(name='encoder_grad_norm')
        self.metrics['planner_grad_norm'] = tf.keras.metrics.Mean(name='planner_grad_norm')

        self.metrics['global_grad_norm'] = tf.keras.metrics.Mean(name='global_grad_norm')

        self.metrics['train_act_with_enc_loss'] = tf.keras.metrics.Mean(name='train_act_with_enc_loss')
        self.metrics['train_act_with_plan_loss'] = tf.keras.metrics.Mean(name='train_act_with_plan_loss')
        self.metrics['valid_act_with_enc_loss'] = tf.keras.metrics.Mean(name='valid_act_with_enc_loss')
        self.metrics['valid_act_with_plan_loss'] = tf.keras.metrics.Mean(name='valid_act_with_plan_loss')

        self.metrics['train_reg_loss'] = tf.keras.metrics.Mean(name='reg_loss')
        self.metrics['valid_reg_loss'] = tf.keras.metrics.Mean(name='valid_reg_loss')

        self.metrics['valid_position_loss'] = tf.keras.metrics.Mean(name='valid_position_loss')
        self.metrics['valid_max_position_loss'] = lfp.metric.MaxMetric(name='valid_max_position_loss')
        self.metrics['valid_rotation_loss'] = tf.keras.metrics.Mean(name='valid_rotation_loss')
        self.metrics['valid_max_rotation_loss'] = lfp.metric.MaxMetric(name='valid_max_rotation_loss')
        self.metrics['valid_gripper_loss'] = tf.keras.metrics.Mean(name='valid_rotation_loss')

        self.chkpt_manager = None

    def compute_loss(self, labels, predictions, mask, seq_lens, weightings=None):
        if self.args.num_distribs is not None:
            per_example_loss = self.nll_action_loss(labels, predictions) * mask
        else:
            per_example_loss = self.mae_action_loss(labels, predictions) * mask

        per_example_loss = tf.reduce_sum(per_example_loss, axis=1) / seq_lens  # take mean along the timestep
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=self.global_batch_size)


    def compute_MAE(self, labels, predictions, mask, seq_lens, weightings=None):
        per_example_loss = self.mae_action_loss(labels, predictions) * mask
        per_example_loss = tf.reduce_sum(per_example_loss, axis=1) / seq_lens  # take mean along the timestep
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=self.global_batch_size)

    def compute_regularisation_loss(self, plan, encoding):
        # Reverse KL(enc|plan): we want planner to map to encoder (weighted by encoder)
        reg_loss = tfd.kl_divergence(encoding, plan)  # + KL(plan, encoding)
        return tf.nn.compute_average_loss(reg_loss, global_batch_size=self.global_batch_size)

    @staticmethod
    def compute_fp16_grads(optimizer, loss, tape, model):
        scaled_loss = optimizer.get_scaled_loss(loss)
        return tape.gradient(scaled_loss, model.trainable_variables)

    def make_sequences_variable_length(self, batch):
        '''
        This is a rather gross tiling/casting/indexing function - but it very effectively vectorises 
        variable sequence lengths over entire batches rather than in the dataloader, which should speed
        us up a lot while retaining the data aug!
        '''
    
        B = batch['obs'].shape[0]
        # Create a variable seq lens tensor
        seq_lens = tf.random.uniform(shape=[B], minval=self.args.window_size_min, 
                                                maxval=self.args.window_size_max, dtype=tf.int32)
        batch['seq_lens'] = tf.cast(seq_lens, tf.float32) #must be a float for later loss per timestep calcs
        # Create a mask which is length variable seq lens 
        mask = tf.cast(tf.sequence_mask(seq_lens, maxlen=self.args.window_size_max), tf.float32) #  B,T mask
        multiply_mask = tf.expand_dims(mask, -1) # B, T, 1 (for broadcasting)

        batch['masks'] = mask
        batch['obs'] *= multiply_mask
        batch['acts'] *= multiply_mask
        # Save goals for later as it depends whether it is imgs or not
        # Get numbers 0>B to concat with the seq lens for gather_nd
        B_range = tf.range(0, B, delta=1, dtype=tf.int32, name='range') # B
        B_indices = tf.stack([B_range, seq_lens], axis = 1) # B,2
        if self.args.images:
            # get the goals corresponding to each of the seq len ends
            goals = tf.gather_nd(batch['imgs'], B_indices)[:, tf.newaxis,:] # B, 1, imgheight, imgwidth, 3
            tile_dims = tf.constant([1, self.args.window_size_max, 1,1,1], tf.int32) 
            goals = tf.tile(goals, tile_dims) # B, T, imgheight, imgwidth, 3
            imgs_mask = tf.cast(multiply_mask, tf.uint8)[:,:,:, tf.newaxis, tf.newaxis] # Must be 5 dim because the imgs are B, T, H, W, C
            batch['goal_imgs'] = goals *imgs_mask
            # End goal specific stuff, start img specific stuff
            batch['imgs'] *= imgs_mask # must be cast as int or this will be SLOW as it converts img to float
            batch['proprioceptive_features'] *= multiply_mask
        else:
            goals = tf.gather_nd(batch['goals'], B_indices)[:, tf.newaxis,:] # B, 1, achieved_goal_dim
            tile_dims = tf.constant([1, self.args.window_size_max, 1], tf.int32) 
            goals = tf.tile(goals, tile_dims) # B, T, achieved_goal_dim
            batch['goals'] *= multiply_mask# B, T, achieved_goal_dim

        return batch


    def step(self, inputs):
        '''
        A function which wraps the shared processing between train and test step
        '''
        states, actions,  goals = inputs['obs'], inputs['acts'], inputs['goals']

        # 1. When using imagesChange the definition of obs_dim to feature encoder dim + proprioceptive features
        # 2. Reshape imgs to B*T H W C.
        # 3. Sub in for states and goals.
        # 4. THen there should be no further changes!
        if self.args.images:
            imgs, proprioceptive_features, goal_imgs = inputs['imgs'], inputs['proprioceptive_features'], inputs['goal_imgs']
            B, T, H, W, C = imgs.shape
            imgs, goal_imgs = tf.reshape(imgs, [B * T, H, W, C]), tf.reshape(goal_imgs, [B * T, H, W, C])
            img_embeddings, goal_embeddings = tf.reshape(self.cnn(imgs), [B, T, -1]), tf.reshape(self.cnn(goal_imgs),[B, T, -1])
            states = tf.concat([img_embeddings, proprioceptive_features],
                               -1)  # gets both the image and it's own xyz ori and angle as pose
            goals = goal_embeddings  # should be B,T, embed_size

        if self.args.gcbc:
            distrib = self.actor([states, goals])
            return distrib
        else:
            encoding = self.encoder([states, actions])
            plan = self.planner([states[:, 0, :], goals[:, 0,
                                                  :]])  # the final goals are tiled out over the entire non masked sequence, so the first timestep is the final goal.
            z_enc = encoding.sample()
            z_plan = plan.sample()
            z_enc_tiled = tf.tile(tf.expand_dims(z_enc, 1), (1, self.dl.window_size, 1))
            z_plan_tiled = tf.tile(tf.expand_dims(z_plan, 1), (1, self.dl.window_size, 1))

            enc_policy = self.actor([states, z_enc_tiled, goals])
            plan_policy = self.actor([states, z_plan_tiled, goals])
            return enc_policy, plan_policy, encoding, plan


    def train_step(self, inputs, beta):
        inputs = self.make_sequences_variable_length(inputs) 

        with tf.GradientTape() as actor_tape, tf.GradientTape() as encoder_tape, tf.GradientTape() as planner_tape:
            actions, seq_lens, mask = inputs['acts'], inputs['seq_lens'], inputs['masks']

            if self.args.gcbc:
                policy = self.step(inputs)
                loss = self.compute_loss(actions, policy, mask, seq_lens)
                gradients = actor_tape.gradient(loss, self.actor.trainable_variables)
                self.actor_optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables))
            else:
                enc_policy, plan_policy, encoding, plan = self.step(inputs)
                act_enc_loss = record(self.compute_loss(actions, enc_policy, mask, seq_lens), self.metrics['train_act_with_enc_loss'])
                act_plan_loss = record(self.compute_loss(actions, plan_policy, mask, seq_lens), self.metrics['train_act_with_plan_loss'])
                reg_loss = record(self.compute_regularisation_loss(plan, encoding), self.metrics['train_reg_loss'])
                loss = act_enc_loss + reg_loss * beta

                if self.args.fp16:
                    actor_gradients = self.compute_fp16_grads(self.actor_optimizer, loss, actor_tape, self.actor)
                    encoder_gradients = self.compute_fp16_grads(self.encoder_optimizer, loss, encoder_tape, self.encoder)
                    planner_gradients = self.compute_fp16_grads(self.planner_optimizer, loss, planner_tape, self.planner)
                else:
                    actor_gradients = actor_tape.gradient(loss, self.actor.trainable_variables)
                    encoder_gradients = encoder_tape.gradient(loss, self.encoder.trainable_variables)
                    planner_gradients = planner_tape.gradient(loss, self.planner.trainable_variables)

                actor_norm = record(tf.linalg.global_norm(actor_gradients), self.metrics['actor_grad_norm'])
                encoder_norm = record(tf.linalg.global_norm(encoder_gradients), self.metrics['encoder_grad_norm'])
                planner_norm = record(tf.linalg.global_norm(planner_gradients), self.metrics['planner_grad_norm'])

                gradients = actor_gradients + encoder_gradients + planner_gradients
                record(tf.linalg.global_norm(gradients), self.metrics['global_grad_norm'])

                self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
                self.encoder_optimizer.apply_gradients(zip(encoder_gradients, self.encoder.trainable_variables))
                self.planner_optimizer.apply_gradients(zip(planner_gradients, self.planner.trainable_variables))


        return record(loss, self.metrics['train_loss'])


    def test_step(self, inputs, beta):
        inputs = self.make_sequences_variable_length(inputs) 
        actions, seq_lens, mask = inputs['acts'], inputs['seq_lens'], inputs['masks']

        if self.args.gcbc:
            policy = self.step(inputs)
            loss = self.compute_loss(actions, policy, mask, seq_lens)
            log_action_breakdown(policy, actions, mask, seq_lens, self.args.num_distribs is not None, self.dl.quaternion_act, self.valid_position_loss, self.valid_max_position_loss, \
                                 self.valid_rotation_loss, self.valid_max_rotation_loss, self.valid_gripper_loss, self.compute_MAE)
        else:
            enc_policy, plan_policy, encoding, plan = self.step(inputs)
            act_enc_loss = record(self.compute_loss(actions, enc_policy, mask, seq_lens), self.metrics['valid_act_with_enc_loss'])
            act_plan_loss = record(self.compute_loss(actions, plan_policy, mask, seq_lens), self.metrics['valid_act_with_plan_loss'])
            reg_loss = record(self.compute_regularisation_loss(plan, encoding), self.metrics['valid_reg_loss'])
            loss = act_plan_loss + reg_loss * beta
            log_action_breakdown(plan_policy, actions, mask, seq_lens, self.args.num_distribs is not None, self.dl.quaternion_act, self.metrics['valid_position_loss'], \
                                 self.metrics['valid_max_position_loss'], self.metrics['valid_rotation_loss'], self.metrics['valid_max_rotation_loss'], self.metrics['valid_gripper_loss'], self.compute_MAE)
        return record(loss,self.metrics['valid_loss'])


    @tf.function
    def distributed_train_step(self, dataset_inputs, beta):
        per_replica_losses = self.strategy.run(self.train_step, args=(dataset_inputs, beta))
        return self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)


    @tf.function
    def distributed_test_step(self, dataset_inputs, beta):
        per_replica_losses = self.strategy.run(self.test_step, args=(dataset_inputs, beta))
        return self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)


    def save_weights(self, path, run_id=None, experiment_key=None):

        if self.args.data_source == 'GCS':
            if self.chkpt_manager is None:
                saved_objects = {'actor': self.actor,
                                 'encoder': self.encoder,
                                 'planner': self.planner,
                                 'actor_optimizer': self.actor_optimizer,
                                 'encoder_optimizer': self.encoder_optimizer,
                                 'planner_optimizer': self.planner_optimizer}
                if self.args.images:
                    ckpt = tf.train.Checkpoint(**saved_objects, cnn=self.cnn)
                else:
                    ckpt = tf.train.Checkpoint(**saved_objects)
                self.chkpt_manager = tf.train.CheckpointManager(ckpt, path, max_to_keep=3)
                save_path = self.chkpt_manager.save()
            else:
                save_path = self.chkpt_manager.save()
        else: # We have to save it all to drive
            os.makedirs(path, exist_ok=True)

            # Save the config as json
            print('Saving training config...')
            with open(f'{path}/config.json', 'w') as f:
                d = vars(self.args)
                d['run_id'] = run_id
                d['experiment_key'] = experiment_key
                d['relative_act'] = self.dl.relative_act
                d['joints'] = self.dl.joints
                d['quaternion_act'] = self.dl.quaternion_act
                json.dump(d, f)

            self.actor.save_weights(f'{path}/actor.h5')
            if not self.args.gcbc:
                self.encoder.save_weights(f'{path}/encoder.h5')
                self.planner.save_weights(f'{path}/planner.h5')
            if self.args.images:
                self.cnn.save_weights(f'{path}/cnn.h5')

            os.makedirs(path+'/optimizers', exist_ok=True)
            np.save(f'{path}/optimizers/actor_optimizer.npy', self.actor_optimizer.get_weights(), allow_pickle=True)
            np.save(f'{path}/optimizers/encoder_optimizer.npy', self.encoder_optimizer.get_weights(), allow_pickle=True)
            np.save(f'{path}/optimizers/planner_optimizer.npy', self.planner_optimizer.get_weights(), allow_pickle=True)


    def load_weights(self, path, with_optimizer=False, from_checkpoint=False):
        # With checkpoint
        if from_checkpoint or self.data_source == 'GCS':
            saved_objects = {'actor': self.actor,
                             'encoder': self.encoder,
                             'planner': self.planner,
                             'actor_optimizer': self.actor_optimizer,
                             'encoder_optimizer': self.encoder_optimizer,
                             'planner_optimizer': self.planner_optimizer}
            if self.args.images:
                ckpt = tf.train.Checkpoint(**saved_objects, cnn=self.cnn)
            else:
                ckpt = tf.train.Checkpoint(**saved_objects)
            self.chkpt_manager = tf.train.CheckpointManager(ckpt, path, max_to_keep=3)
            ckpt.restore(tf.train.latest_checkpoint(path))

        # Without checkpointing, because it was created on GDRIVE
        self.actor.load_weights(f'{path}/actor.h5')
        if not self.args.gcbc:
            self.encoder.load_weights(f'{path}/encoder.h5')
            self.planner.load_weights(f'{path}/planner.h5')
        if self.args.images:
            self.cnn.load_weights(f'{path}/cnn.h5')

        if with_optimizer:
            self.load_optimizer_state(self.actor_optimizer, f'{path}/optimizers/actor_optimizer.npy', self.actor.trainable_variables)
            if not self.args.gcbc:
                self.load_optimizer_state(self.encoder_optimizer, f'{path}/optimizers/encoder_optimizer.npy', self.encoder.trainable_variables)
                self.load_optimizer_state(self.planner_optimizer, f'{path}/optimizers/planner_optimizer.npy', self.planner.trainable_variables)


    def load_optimizer_state(self, optimizer, load_path, trainable_variables):
        def optimizer_step():
            # need to do this to initialize the optimiser
            # dummy zero gradients
            zero_grads = [tf.zeros_like(w) for w in trainable_variables]
            # save current state of variables
            saved_vars = [tf.identity(w) for w in trainable_variables]

            # Apply gradients which don't do anything
            optimizer.apply_gradients(zip(zero_grads, trainable_variables))

            # Reload variables
            [x.assign(y) for x, y in zip(trainable_variables, saved_vars)]
            return 0.0

        @tf.function
        def distributed_opt_step():
            '''
            Only used for optimizer checkpointing - we need to run a pass to initialise all the optimizer weights. Can't use restore as colab TPUs don't have a local filesystem.
            '''
            per_replica_losses = self.strategy.run(optimizer_step, args=())
            return self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)

        # Load optimizer weights
        opt_weights = np.load(load_path, allow_pickle=True)

        # init the optimiser
        distributed_opt_step()
        # Set the weights of the optimizer
        optimizer.set_weights(opt_weights)
