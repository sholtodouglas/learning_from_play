import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import Progbar
import tensorflow_addons as tfa
from tensorflow_addons.optimizers import AdamW
import lfp
import os

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

    # Loss functions
    nll_action_loss = lambda y, p_y: tf.reduce_sum(-p_y.log_prob(y), axis=2)
    mae_action_loss = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
    mse_action_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    kl_loss = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)

    def __init__(self, actor, window_size, quaternion_act, probabilistic, batch_size,
                 encoder=None, planner=None, distribute_strategy=None, learning_rate='1e-4', gcbc=False):
        self.actor = actor
        self.encoder = encoder
        self.planner = planner
        self.distribute_strategy = distribute_strategy
        self.optimizer = Adam(learning_rate)
        self.gcbc = gcbc
        self.window_size = window_size
        self.quaternion_act = quaternion_act
        self.probabilistic = probabilistic
        self.batch_size = batch_size

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
        reg_loss = self.kl_loss(encoding, plan)
        return tf.nn.compute_average_loss(reg_loss, global_batch_size=self.batch_size)


    # Now outside strategy .scope
    def train_step(self, inputs, beta, prev_global_grad_norm):
        with tf.GradientTape() as tape:  # separate planner tape for INFO VAE
            # Todo: figure out mask and seq_lens for new dataset
            states, actions, goals, seq_lens, mask = inputs['obs'], inputs['acts'], inputs['goals'], inputs['seq_lens'], \
                                                     inputs['masks']
            if self.gcbc:
                distrib = self.actor([states, goals])
                loss = self.compute_loss(actions, distrib, mask, seq_lens)
                gradients = tape.gradient(loss, self.actor.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables))
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
                act_loss = act_enc_loss

                reg_loss = self.compute_regularisation_loss(plan, encoding)
                train_reg_loss.update_state(reg_loss)
                train_act_with_enc_loss.update_state(act_enc_loss)
                train_act_with_plan_loss.update_state(act_plan_loss)

                loss = act_loss + reg_loss * beta

                gradients = tape.gradient(loss, trainable_variables)
                actor_norm = tf.linalg.global_norm(gradients[:actor_grad_len])
                encoder_norm = tf.linalg.global_norm(gradients[actor_grad_len:actor_grad_len + encoder_grad_len])
                planner_norm = tf.linalg.global_norm(
                    gradients[
                    actor_grad_len + encoder_grad_len:actor_grad_len + encoder_grad_len + planner_grad_len])
                actor_grad_norm.update_state(actor_norm)
                encoder_grad_norm.update_state(encoder_norm)
                planner_grad_norm.update_state(planner_norm)

                # scale planner gradients

                # if the gradient norm is more than 3x the previous one, clip it to the previous norm for stability
                gradients = tf.cond(tf.linalg.global_norm(gradients) > 3 * prev_global_grad_norm,
                                    lambda: tf.clip_by_global_norm(gradients, prev_global_grad_norm)[0],
                                    lambda: gradients)  # must get[0] as it returns new norm as [1]
                # make the planner converge more quickly
                planner_grads = gradients[
                                actor_grad_len + encoder_grad_len:actor_grad_len + encoder_grad_len + planner_grad_len]

                planner_grads = [g * 10 for g in planner_grads]
                gradients = gradients[:actor_grad_len] + gradients[
                                                         actor_grad_len:actor_grad_len + encoder_grad_len] + planner_grads

                actor_norm_clipped = tf.linalg.global_norm(gradients[:actor_grad_len])
                encoder_norm_clipped = tf.linalg.global_norm(
                    gradients[actor_grad_len:actor_grad_len + encoder_grad_len])
                planner_norm_clipped = tf.linalg.global_norm(
                    gradients[
                    actor_grad_len + encoder_grad_len:actor_grad_len + encoder_grad_len + planner_grad_len])
                actor_grad_norm_clipped.update_state(actor_norm_clipped)
                encoder_grad_norm_clipped.update_state(encoder_norm_clipped)
                planner_grad_norm_clipped.update_state(planner_norm_clipped)

                global_grad_norm.update_state(tf.linalg.global_norm(gradients))

                self.optimizer.apply_gradients(zip(gradients,
                                              self.actor.trainable_variables + \
                                              self.encoder.trainable_variables + \
                                              self.planner.trainable_variables))
        train_loss.update_state(loss)

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
            valid_reg_loss.update_state(reg_loss)

            valid_act_with_enc_loss.update_state(act_enc_loss)
            valid_act_with_plan_loss.update_state(act_plan_loss)

            # pos, rot, gripper individual losses
            if self.probabilistic:
                pos_acts, rot_acts, grip_act = tf.split(plan_policy.sample(), action_breakdown, -1)
            else:
                pos_acts, rot_acts, grip_act = tf.split(plan_policy, action_breakdown, -1)

            loss = act_loss + reg_loss * beta

        true_pos_acts, true_rot_acts, true_grip_act = tf.split(actions, action_breakdown, -1)
        valid_position_loss.update_state(compute_MAE(true_pos_acts, pos_acts, mask, seq_lens))
        valid_max_position_loss(true_pos_acts, pos_acts, mask)
        valid_rotation_loss.update_state(compute_MAE(true_rot_acts, rot_acts, mask, seq_lens))
        valid_max_rotation_loss(true_rot_acts, rot_acts, mask)
        valid_gripper_loss.update_state(compute_MAE(true_grip_act, grip_act, mask, seq_lens))
        valid_loss.update_state(loss)

        if self.gcbc:
            return loss
        else:
            return loss, z_enc, z_plan

    @tf.function
    def distributed_train_step(self, dataset_inputs, beta, prev_global_grad_norm):
        per_replica_losses = strategy.run(self.train_step, args=(dataset_inputs, beta, prev_global_grad_norm))
        return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)

    @tf.function
    def distributed_test_step(self, dataset_inputs, beta):
        if self.gcbc:
            per_replica_losses = strategy.run(self.test_step, args=(dataset_inputs, beta))
            return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)
        else:
            per_replica_losses, ze, zp = strategy.run(self.test_step, args=(dataset_inputs, beta))
            return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None), ze.values[0], zp.values[
                0]

    # TODO: add dataloader to input args - refactor other input args for window size etc.
    # TODO: complete all following methods
    def save_weights(path, actor, encoder=None, planner=None, step=""):
        os.makedirs(path, exist_ok=True)

        print('Saving model weights...')
        # Save the standardisation params
        np.savez(path + 'hyper_params', relative_obs=dataloader.relative_obs, relative_act=dataloader.relative_act,
                 quaternion_act=dataloader.quaternion_act,
                 joints=dataloader.joints, LAYER_SIZE=LAYER_SIZE, LATENT_DIM=LATENT_DIM, GRIPPER_WEIGHT=GRIPPER_WEIGHT,
                 GCBC=GCBC, PROBABILISTIC=PROBABILISTIC, QUANTISED=QUANTISED, run_id=wandb.run.id, scaling=scaling,
                 N_QUANTISATIONS=N_QUANTISATIONS)
        # save timestepped version
        if step != "":
            actor.save_weights(path + 'model_' + str(step) + '.h5')
            if planner is not None: planner.save_weights(path + 'planner_' + str(step) + '.h5')
            if encoder is not None: encoder.save_weights(path + 'encoder_' + str(step) + '.h5')

        # save the latest version
        actor.save_weights(path + 'model.h5')
        if planner is not None: planner.save_weights(path + 'planner.h5')
        if encoder is not None: encoder.save_weights(path + 'encoder.h5')

        # save the optimizer state
        np.save(os.path.join(path, 'optimizer'), optimizer.get_weights())

    def load_weights(path, actor, encoder=None, planner=None, step=""):
        actor.load_weights(f'{path}/model' + step + '.h5')
        if planner is not None: planner.load_weights(f'{path}/planner' + step + '.h5')
        if encoder is not None: encoder.load_weights(f'{path}/encoder' + step + '.h5')

    def load_optimizer_state(optimizer, load_path):
        # Load optimizer weights
        opt_weights = np.load(load_path + 'optimizer.npy', allow_pickle=True)

        # init the optimiser
        distributed_opt_step()
        # Set the weights of the optimizer
        optimizer.set_weights(opt_weights)

    def optimizer_step():
        # need to do this to initialize the optimiser
        model_train_vars = actor.trainable_variables + encoder.trainable_variables + planner.trainable_variables
        # dummy zero gradients
        zero_grads = [tf.zeros_like(w) for w in model_train_vars]
        # save current state of variables
        saved_vars = [tf.identity(w) for w in model_train_vars]

        # Apply gradients which don't do anything
        optimizer.apply_gradients(zip(zero_grads, model_train_vars))

        # Reload variables
        [x.assign(y) for x, y in zip(model_train_vars, saved_vars)]
        return 0.0

    @tf.function
    def distributed_opt_step():
        '''
        Only used for optimizer checkpointing - we need to run a pass to initialise all the optimizer weights. Can't use restore as colab TPUs don't have a local filesystem.
        '''
        per_replica_losses = strategy.run(optimizer_step, args=())
        return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)

