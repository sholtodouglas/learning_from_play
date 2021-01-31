import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import Progbar
from tensorflow.distribute import ReduceOp
import tensorflow_probability as tfp
import tensorflow_addons as tfa
from tensorflow_addons.optimizers import AdamW
import lfp
import os
import wandb
import json

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

    # Losses
    nll_action_loss = lambda y, p_y: tf.reduce_sum(-p_y.log_prob(y), axis=2)
    mae_action_loss = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
    mse_action_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

    def __init__(self, dataloader, actor, probabilistic, encoder=None, planner=None,
                 distribute_strategy=None, learning_rate='3e-4', plan_lr_multiplier=10, clipnorm=1.0, gcbc=False):
        self.actor = actor
        self.encoder = encoder
        self.planner = planner
        self.distribute_strategy = distribute_strategy
        self.probabilistic = probabilistic
        self.gcbc = gcbc
        self.window_size = dataloader.window_size
        self.quaternion_act = dataloader.quaternion_act
        self.batch_size = dataloader.batch_size

        self.actor_optimizer = Adam(learning_rate=learning_rate, global_clipnorm=clipnorm)
        self.encoder_optimizer = Adam(learning_rate=learning_rate, global_clipnorm=clipnorm)
        self.planner_optimizer = Adam(learning_rate=plan_lr_multiplier*learning_rate, global_clipnorm=clipnorm)

        # Metrics
        self.metrics = {}
        self.metrics['train_loss'] = tf.keras.metrics.Mean(name='train_loss')
        self.metrics['actor_grad_norm'] = tf.keras.metrics.Mean(name='actor_grad_norm')
        self.metrics['valid_loss'] = tf.keras.metrics.Mean(name='valid_loss')
        self.metrics['valid_position_loss'] = tf.keras.metrics.Mean(name='valid_position_loss')
        self.metrics['valid_max_position_loss'] = lfp.metric.MaxMetric(name='valid_max_position_loss')
        self.metrics['valid_rotation_loss'] = tf.keras.metrics.Mean(name='valid_rotation_loss')
        self.metrics['valid_max_rotation_loss'] = lfp.metric.MaxMetric(name='valid_max_rotation_loss')
        self.metrics['valid_gripper_loss'] = tf.keras.metrics.Mean(name='valid_gripper_loss')
        if not self.gcbc:
            self.metrics['train_reg_loss'] = tf.keras.metrics.Mean(name='train_reg_loss')
            self.metrics['train_act_with_enc_loss'] = tf.keras.metrics.Mean(name='train_act_with_enc_loss')
            self.metrics['train_act_with_plan_loss'] = tf.keras.metrics.Mean(name='train_act_with_plan_loss')
            self.metrics['encoder_grad_norm'] = tf.keras.metrics.Mean(name='encoder_grad_norm')
            self.metrics['planner_grad_norm'] = tf.keras.metrics.Mean(name='planner_grad_norm')
            self.metrics['valid_reg_loss'] = tf.keras.metrics.Mean(name='valid_reg_loss')
            self.metrics['valid_act_with_enc_loss'] = tf.keras.metrics.Mean(name='valid_act_with_enc_loss')
            self.metrics['valid_act_with_plan_loss'] = tf.keras.metrics.Mean(name='valid_act_with_plan_loss')

    def compute_loss(self, labels, predictions, mask, seq_lens):
        if self.probabilistic:
            per_example_loss = self.nll_action_loss(labels, predictions) * mask
        else:
            per_example_loss = self.mae_action_loss(labels, predictions) * mask

        per_example_loss = tf.reduce_sum(per_example_loss, axis=1) / seq_lens  # take mean along the timestep
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=self.batch_size)


    def compute_MAE(self, labels, predictions, mask, seq_lens, weightings=None):
        per_example_loss = self.mae_action_loss(labels, predictions) * mask
        per_example_loss = tf.reduce_sum(per_example_loss, axis=1) / seq_lens  # take mean along the timestep
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=self.batch_size)


    def compute_regularisation_loss(self, plan, encoding):
        # Reverse KL(enc|plan): we want planner to map to encoder (weighted by encoder)
        reg_loss = tfp.distributions.kl_divergence(encoding, plan)
        return tf.nn.compute_average_loss(reg_loss, global_batch_size=self.batch_size)


    # Now outside strategy .scope
    def train_step(self, inputs, beta):
        # Todo: figure out mask and seq_lens for new dataset
        states, actions, goals, seq_lens, mask = inputs['obs'], inputs['acts'], inputs['goals'], inputs['seq_lens'], \
                                                 inputs['masks']
        if self.gcbc:
            with tf.GradientTape() as actor_tape:
                distrib = self.actor([states, goals])
                loss = self.compute_loss(actions, distrib, mask, seq_lens)
                gradients = actor_tape.gradient(loss, self.actor.trainable_variables)
                self.actor_optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables))
        else:
            with tf.GradientTape() as actor_tape, tf.GradientTape() as encoder_tape, tf.GradientTape() as planner_tape:
                encoding = self.encoder([states, actions])
                plan = self.planner([states[:, 0, :], goals[:, 0, :]])  # the final goals are tiled out over the entire non masked sequence, so the first timestep is the final goal.
                z_enc = encoding.sample()
                z_plan = plan.sample()
                z_enc_tiled = tf.tile(tf.expand_dims(z_enc, 1), (1, self.window_size, 1))
                z_plan_tiled = tf.tile(tf.expand_dims(z_plan, 1), (1, self.window_size, 1))

                enc_policy = self.actor([states, z_enc_tiled, goals])
                plan_policy = self.actor([states, z_plan_tiled, goals])

                act_enc_loss = self.compute_loss(actions, enc_policy, mask, seq_lens)
                act_plan_loss = self.compute_loss(actions, plan_policy, mask, seq_lens)
                act_loss = act_enc_loss

                reg_loss = self.compute_regularisation_loss(plan, encoding)

                loss = act_loss + reg_loss * beta

                # Gradients
                actor_gradients = actor_tape.gradient(loss, self.actor.trainable_variables)
                encoder_gradients = encoder_tape.gradient(loss, self.encoder.trainable_variables)
                planner_gradients = planner_tape.gradient(loss, self.planner.trainable_variables)
                all_gradients = actor_gradients + encoder_gradients + planner_gradients # concat lists

                # Gradient norms
                actor_norm = tf.linalg.global_norm(actor_gradients)
                encoder_norm = tf.linalg.global_norm(encoder_gradients)
                planner_norm = tf.linalg.global_norm(planner_gradients)
                # global_norm = tf.linalg.global_norm(all_gradients)

                # # scale gradients
                # def clip_gradients(gradients, gradient_norm, prev_gradient_norm, max_blowup=3):
                #     # if the gradient norm is more than 3x the previous one, clip it to the previous norm for stability
                #     gradients = tf.cond(gradient_norm > max_blowup * prev_gradient_norm,
                #                         lambda: tf.clip_by_global_norm(gradients, prev_global_grad_norm)[0],
                #                         lambda: gradients)  # must get[0] as it returns new norm as [1]
                #     return gradients

                # make the planner converge more quickly
                # actor_gradients = clip_gradients(actor_gradients, )


                # actor_grad_norm_clipped.update_state(actor_norm_clipped)
                # encoder_grad_norm_clipped.update_state(encoder_norm_clipped)
                # planner_grad_norm_clipped.update_state(planner_norm_clipped)

                # Optimizer step
                self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
                self.encoder_optimizer.apply_gradients(zip(encoder_gradients, self.encoder.trainable_variables))
                self.planner_optimizer.apply_gradients(zip(planner_gradients, self.planner.trainable_variables))

                # Train Metrics
                self.metrics['train_reg_loss'].update_state(reg_loss)
                self.metrics['train_act_with_enc_loss'].update_state(act_enc_loss)
                self.metrics['train_act_with_plan_loss'].update_state(act_plan_loss)

                self.metrics['actor_grad_norm'].update_state(actor_norm)
                self.metrics['encoder_grad_norm'].update_state(encoder_norm)
                self.metrics['planner_grad_norm'].update_state(planner_norm)
        self.metrics['train_loss'].update_state(loss)

        return loss

    def test_step(self, inputs, beta):
        states, actions, goals, seq_lens, mask = inputs['obs'], inputs['acts'], inputs['goals'], inputs['seq_lens'], \
                                                 inputs['masks']
        if self.quaternion_act:
            # xyz, q1-4, grip
            action_breakdown = [3, 4, 1]
        else:
            action_breakdown = [3, 3, 1]

        if self.gcbc:
            policy = self.actor([states, goals], training=False)
            loss = self.compute_loss(actions, policy, mask, seq_lens)
            if self.probabilistic:
                pos_acts, rot_acts, grip_act = tf.split(policy.sample(), action_breakdown, -1)
            else:
                pos_acts, rot_acts, grip_act = tf.split(policy, action_breakdown, -1)
        else:
            encoding = self.encoder([states, actions])
            plan = self.planner([states[:, 0, :], goals[:, 0, :]])  # the final goals are tiled out over the entire non masked sequence, so the first timestep is the final goal.
            z_enc = encoding.sample()
            z_plan = plan.sample()
            z_enc_tiled = tf.tile(tf.expand_dims(z_enc, 1), (1, self.window_size, 1))
            z_plan_tiled = tf.tile(tf.expand_dims(z_plan, 1), (1, self.window_size, 1))

            enc_policy = self.actor([states, z_enc_tiled, goals])
            plan_policy = self.actor([states, z_plan_tiled, goals])

            act_enc_loss = self.compute_loss(actions, enc_policy, mask, seq_lens)
            act_plan_loss = self.compute_loss(actions, plan_policy, mask, seq_lens)
            act_loss = act_plan_loss

            reg_loss = self.compute_regularisation_loss(plan, encoding)

            # pos, rot, gripper individual losses
            if self.probabilistic:
                pos_acts, rot_acts, grip_act = tf.split(plan_policy.sample(), action_breakdown, -1)
            else:
                pos_acts, rot_acts, grip_act = tf.split(plan_policy, action_breakdown, -1)

            loss = act_loss + reg_loss * beta

        true_pos_acts, true_rot_acts, true_grip_act = tf.split(actions, action_breakdown, -1)

        # Validation Metrics
        self.metrics['valid_reg_loss'].update_state(reg_loss)
        self.metrics['valid_act_with_enc_loss'].update_state(act_enc_loss)
        self.metrics['valid_act_with_plan_loss'].update_state(act_plan_loss)
        self.metrics['valid_position_loss'].update_state(self.compute_MAE(true_pos_acts, pos_acts, mask, seq_lens))
        self.metrics['valid_max_position_loss'](true_pos_acts, pos_acts, mask)
        self.metrics['valid_rotation_loss'].update_state(self.compute_MAE(true_rot_acts, rot_acts, mask, seq_lens))
        self.metrics['valid_max_rotation_loss'](true_rot_acts, rot_acts, mask)
        self.metrics['valid_gripper_loss'].update_state(self.compute_MAE(true_grip_act, grip_act, mask, seq_lens))
        self.metrics['valid_loss'].update_state(loss)

        # Results + clear state for all metrics in metrics dict
        metric_results = {metric_name: lfp.metric.log(metric) for metric_name, metric in self.metrics.items()}
        metric_results['beta'] = beta

        if self.gcbc:
            return loss, metric_results
        else:
            return loss, metric_results, z_enc, z_plan

    @tf.function
    def distributed_train_step(self, dataset_inputs, beta):
        per_replica_losses = self.distribute_strategy.run(self.train_step, args=(dataset_inputs, beta))
        losses = self.distribute_strategy.reduce(ReduceOp.MEAN, per_replica_losses, axis=None)
        return losses

    @tf.function
    def distributed_test_step(self, dataset_inputs, beta):
        if self.gcbc:
            per_replica_losses, metrics = self.distribute_strategy.run(self.test_step, args=(dataset_inputs, beta))
            losses = self.distribute_strategy.reduce(ReduceOp.MEAN, per_replica_losses, axis=None)
            metrics = {k: self.distribute_strategy.reduce(ReduceOp.MEAN, v, axis=None) for k, v in metrics.items()}
            return losses, metrics
        else:
            per_replica_losses, metrics, ze, zp = self.distribute_strategy.run(self.test_step,
                                                                               args=(dataset_inputs, beta))
            losses = self.distribute_strategy.reduce(ReduceOp.MEAN, per_replica_losses, axis=None)
            metrics = {k: self.distribute_strategy.reduce(ReduceOp.MEAN, v, axis=None) for k, v in metrics.items()}
            return losses, metrics, ze.values[0], zp.values[0]

    def save_weights(self, path, config=None, step=""):
        os.makedirs(path, exist_ok=True)

        # Save the config as json
        if config is not None:
            print('Saving training config...')
            with open(f'{path}/config.json', 'w') as f:
                json.dump(vars(config), f)

        # save timestepped version
        print('Saving model weights...')
        if step != "":
            self.actor.save_weights(f'{path}/actor_{str(step)}.h5')
            if not self.gcbc:
                self.encoder.save_weights(f'{path}/encoder_{str(step)}.h5')
                self.planner.save_weights(f'{path}/planner_{str(step)}.h5')

        # save the latest version
        self.actor.save_weights(f'{path}/actor.h5')
        if not self.gcbc:
            self.encoder.save_weights(f'{path}/encoder.h5')
            self.planner.save_weights(f'{path}/planner.h5')

        # save the optimizer state
        np.save(f'{path}/actor_optimizer.npy', self.actor_optimizer.get_weights())
        if not self.gcbc:
            np.save(f'{path}/encoder_optimizer.npy', self.encoder_optimizer.get_weights())
            np.save(f'{path}/planner_optimizer.npy', self.planner_optimizer.get_weights())

    def load_weights(self, path, with_optimizer=False, step=""):
        self.actor.load_weights(f'{path}/actor_{str(step)}.h5')
        if not self.gcbc:
            self.encoder.load_weights(f'{path}/encoder_{str(step)}.h5')
            self.planner.load_weights(f'{path}/planner_{str(step)}.h5')
            
        if with_optimizer:
            self.load_optimizer_state(self.actor_optimizer, f'{path}/optimizers/actor_optimizer.npy')
            if not self.gcbc:
                self.load_optimizer_state(self.encoder_optimizer, f'{path}/optimizers/encoder_optimizer.npy')
                self.load_optimizer_state(self.planner_optimizer, f'{path}/optimizers/planner_optimizer.npy')

    @staticmethod
    def load_optimizer_state(optimizer, load_path, strategy, trainable_variables):
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
            per_replica_losses = strategy.run(optimizer_step, args=())
            return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)

        # Load optimizer weights
        opt_weights = np.load(load_path, allow_pickle=True)

        # init the optimiser
        distributed_opt_step()
        # Set the weights of the optimizer
        optimizer.set_weights(opt_weights)