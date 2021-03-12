import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU, Input, LSTM, Concatenate, Masking, Reshape, Lambda, \
    Bidirectional, GRU, LayerNormalization, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.metrics import Mean
from tensorflow.python.ops.gen_linalg_ops import SelfAdjointEig
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
tfpl = tfp.layers
from lfp.metric import MaxMetric

ACT_LIMITS = tf.constant([1.5, 1.5, 2.2, 3.2, 3.2, 3.2, 1.1])

def create_actor(obs_dim, act_dim, goal_dim, layer_size=1024, vocab_size=1024, training=True):
    # params #
    batch_size = None if training else 1
    stateful = not training

    # Input #
    o = Input(shape=(None, obs_dim), batch_size=batch_size, dtype=tf.float32, name='input_obs')
    z = Input(shape=(None, vocab_size), batch_size=batch_size, dtype=tf.float32, name='input_vocab')
    g = Input(shape=(None, goal_dim), batch_size=batch_size, dtype=tf.float32, name='input_goals')

    # RNN #
    x = Concatenate(axis=-1)([o, z, g])
    x = Masking(mask_value=0.)(x)
    x = LSTM(layer_size, return_sequences=True, stateful=stateful, name='LSTM_in_1')(x)
    x = LSTM(layer_size, return_sequences=True, stateful=stateful, name='LSTM_in_2')(x)

    # Deterministic output #
    actions = Dense(act_dim, activation='tanh', name='acts')(x)
    actions = Lambda(lambda a: a * ACT_LIMITS)(actions) # scale to action limits
    return Model([o, z, g], actions)


def create_encoder(obs_dim, act_dim, layer_size=2048, vocab_size=1024):
    # Input #
    obs = Input(shape=(None, obs_dim), dtype=tf.float32, name='obs')
    acts = Input(shape=(None, act_dim), dtype=tf.float32, name='acts')

    # Layers #
    x = Concatenate(axis=-1)([obs, acts])
    x = Masking(mask_value=0.)(x)
    x = Bidirectional(LSTM(layer_size, return_sequences=True), merge_mode='concat')(x)
    x = Bidirectional(LSTM(layer_size, return_sequences=False), merge_mode='concat')(x)

    logits = Dense(vocab_size, name='to_vocab')(x)
    return Model([obs, acts], logits)

    # Latent Variable #
    # x = Dense(tfpl.MultivariateNormalTriL.params_size(latent_dim), activation=None)(x),
    # z = tfpl.MultivariateNormalTriL(latent_dim, name='latent')(x)
    # return Model([obs, acts], z)


def create_planner(obs_dim, goal_dim, layer_size=2048, latent_dim=256):
    # params #
    batch_size = None

    # Input #
    o_i = Input(shape=(obs_dim,), batch_size=batch_size, dtype=tf.float32,
                name='initial_obs')  # has arm state
    o_g = Input(shape=(goal_dim,), batch_size=batch_size, dtype=tf.float32,
                name='goal_obs')  # does not have arm state

    # Layers #
    x = Concatenate(axis=-1)([o_i, o_g])
    x = Masking(mask_value=0.)(x)
    x = Dense(layer_size, activation="relu", name='layer_1')(x) # maybe change to selu/gelu/swish?
    x = Dense(layer_size, activation="relu", name='layer_2')(x)
    x = Dense(layer_size, activation="relu", name='layer_3')(x)
    x = Dense(layer_size, activation="relu", name='layer_4')(x)

    # Latent Variable #
    x = Dense(tfpl.MultivariateNormalTriL.params_size(latent_dim), activation=None)(x),
    z = tfpl.MultivariateNormalTriL(latent_dim, name='latent')(x)
    return Model([o_i, o_g], z)

# Todo: add beta callback, add checkpointing callback, think about train=False autoregressive, what to do about masking?
# Account for probabilistic (need to sample the actions to get MAE)
class LFPNet(Model):
    def __init__(self, encoder, planner, actor, beta, temperature=1/16) -> None:
        super().__init__()
        self.encoder = encoder
        self.planner = planner
        self.actor = actor
        self.beta = beta
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.valid_loss = tf.keras.metrics.Mean(name='valid_loss')
        self.temperature = temperature

        self.train_act_with_enc_loss = tf.keras.metrics.Mean(name='train_act_with_enc_loss')
        self.train_act_with_plan_loss = tf.keras.metrics.Mean(name='train_act_with_plan_loss')
        self.valid_act_with_enc_loss = tf.keras.metrics.Mean(name='valid_act_with_enc_loss')
        self.valid_act_with_plan_loss = tf.keras.metrics.Mean(name='valid_act_with_plan_loss')

        self.train_reg_loss = tf.keras.metrics.Mean(name='reg_loss')
        self.valid_reg_loss = tf.keras.metrics.Mean(name='valid_reg_loss')
        self.beta_metric = tf.keras.metrics.Mean(name='beta')

        self.valid_position_loss = tf.keras.metrics.Mean(name='valid_position_loss')
        self.valid_max_position_loss = MaxMetric(name='valid_max_position_loss')
        self.valid_rotation_loss = tf.keras.metrics.Mean(name='valid_rotation_loss')
        self.valid_max_rotation_loss = MaxMetric(name='valid_max_rotation_loss')
        self.valid_gripper_loss = tf.keras.metrics.Mean(name='valid_rotation_loss')

    def call(self, inputs, planner=True, training=False):

        logits = self.encoder([inputs['obs'], inputs['acts']])

        z_q = tfpl.DistributionLambda(
            lambda logits: tfd.RelaxedOneHotCategorical(self.temperature, logits)
        )(logits)

        z_hard = tf.math.argmax(logits, axis=-1)
        z_hard = tf.one_hot(z_hard, logits.shape[-1], dtype=z_q.dtype)

        z = z_q + tf.stop_gradient(z_hard - z_q)

        # import pdb; pdb.set_trace()

        z_tiled = tf.tile(tf.expand_dims(z, 1), (1, inputs['obs'].shape[1], 1))

        acts = self.actor([inputs['obs'], z_tiled, inputs['goals']])
        return acts, z

        # if planner:
        #     z = self.planner([inputs['obs'][:,0,:], inputs['goals'][:,0,:]])
        # else:
        #     z = self.encoder([inputs['obs'], inputs['acts']])
        # z_tiled = tf.tile(tf.expand_dims(z[0], 1), (1, inputs['obs'].shape[1], 1))
        # acts = self.actor([inputs['obs'], z_tiled, inputs['goals']])
        # return acts, z

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            acts_enc, z_enc = self(inputs, planner=False, training=True)
            acts_plan, z_plan = self(inputs, planner=True, training=True)
            act_enc_loss = self.compiled_loss(inputs['acts'], acts_enc, regularization_losses=self.losses)
            act_plan_loss = self.compiled_loss(inputs['acts'], acts_plan, regularization_losses=self.losses)

            # reg_loss = tfd.kl_divergence(z_enc, z_plan)
            reg_loss = tf.zeros_like(act_enc_loss)
            loss = act_enc_loss + self.beta * reg_loss

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        
        # Update metrics 
        self.train_loss.update_state(loss)
        self.train_act_with_enc_loss.update_state(act_enc_loss)
        self.train_act_with_plan_loss.update_state(act_plan_loss)
        self.train_reg_loss.update_state(reg_loss)
        self.beta_metric.update_state(self.beta)

        result = {m.name: m.result() for m in self.metrics}
        result['beta'] = self.beta
        return result

    def test_step(self, inputs):
        acts_enc, z_enc = self(inputs, planner=False, training=False)
        acts_plan, z_plan = self(inputs, planner=True, training=False)
        act_enc_loss = self.compiled_loss(inputs['acts'], acts_enc, regularization_losses=self.losses)
        act_plan_loss = self.compiled_loss(inputs['acts'], acts_plan, regularization_losses=self.losses)

        # reg_loss = tfd.kl_divergence(z_enc, z_plan)
        reg_loss = tf.zeros_like(act_plan_loss)
        loss = act_plan_loss + self.beta * reg_loss

        # Update metrics 
        self.valid_loss.update_state(loss)
        self.valid_act_with_enc_loss.update_state(act_enc_loss)
        self.valid_act_with_plan_loss.update_state(act_plan_loss)
        self.valid_reg_loss.update_state(reg_loss)

        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [
            self.train_loss,
            self.valid_loss,
            self.train_act_with_enc_loss,
            self.train_act_with_plan_loss,
            self.valid_act_with_enc_loss,
            self.valid_act_with_plan_loss,
            self.train_reg_loss,
            self.valid_reg_loss,
            self.valid_position_loss,
            self.valid_max_position_loss,
            self.valid_rotation_loss,
            self.valid_max_rotation_loss,
            self.valid_gripper_loss,
        ]