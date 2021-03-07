import attr
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU, Input, LSTM, Concatenate, Masking, Reshape, Lambda, \
    Bidirectional, GRU, LayerNormalization, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.metrics import Mean
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
tfpl = tfp.layers

ACT_LIMITS = tf.constant([1.5, 1.5, 2.2, 3.2, 3.2, 3.2, 1.1])

def create_actor(obs_dim, act_dim, goal_dim, layer_size=1024, latent_dim=256, training=True):
    # params #
    batch_size = None if training else 1
    stateful = not training

    # Input #
    o = Input(shape=(None, obs_dim), batch_size=batch_size, dtype=tf.float32, name='input_obs')
    z = Input(shape=(None, latent_dim), batch_size=batch_size, dtype=tf.float32, name='input_latent')
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


def create_encoder(obs_dim, act_dim, layer_size=2048, latent_dim=256):
    # Input #
    obs = Input(shape=(None, obs_dim), dtype=tf.float32, name='obs')
    acts = Input(shape=(None, act_dim), dtype=tf.float32, name='acts')

    # Layers #
    x = Concatenate(axis=-1)([obs, acts])
    x = Masking(mask_value=0.)(x)
    x = Bidirectional(LSTM(layer_size, return_sequences=True), merge_mode='concat')(x)
    x = Bidirectional(LSTM(layer_size, return_sequences=False), merge_mode='concat')(x)

    # Latent Variable #
    x = Dense(tfpl.MultivariateNormalTriL.params_size(latent_dim), activation=None)(x),
    z = tfpl.MultivariateNormalTriL(latent_dim, name='latent')(x)
    return Model([obs, acts], z)


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
@attr.s(eq=False, repr=False)
class LFPNet(Model):
    encoder: Model = attr.ib()
    planner: Model = attr.ib()
    actor: Model = attr.ib()
    beta: float = attr.ib()

    def __attrs_post_init__(self) -> None:
        super(LFPNet, self).__init__()
        self.total_loss_tracker = Mean(name="total_loss")
        self.action_loss_tracker = Mean(name="action_loss")
        self.reg_loss_tracker = Mean(name="reg_loss")

    def call(self, inputs, planner=True, training=False):
        if planner:
            z = self.planner([inputs['obs'][:,0,:], inputs['goals'][:,0,:]])
        else:
            z = self.encoder([inputs['obs'], inputs['acts']])
        z_tiled = tf.tile(tf.expand_dims(z, 1), (1, inputs['obs'].shape[1], 1))
        acts = self.actor([inputs['obs'], z_tiled, inputs['goals']])
        return acts, z

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            acts_enc, z_enc = self(inputs, planner=False, training=True)
            acts_plan, z_plan = self(inputs, planner=True, training=True)
            act_loss = self.compiled_loss(inputs['acts'], acts_enc, regularization_losses=self.losses)
            reg_loss = tfd.kl_divergence(z_enc, z_plan)
            loss = act_loss + self.beta * reg_loss

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # Update metrics (includes the metric that tracks the loss)
        self.total_loss_tracker.update_state(loss)
        self.action_loss_tracker.update_state(act_loss)
        self.reg_loss_tracker.update_state(reg_loss)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, inputs):
        acts_enc, z_enc = self(inputs, planner=False, training=False)
        acts_plan, z_plan = self(inputs, planner=True, training=False)
        act_loss = self.compiled_loss(inputs['acts'], acts_plan, regularization_losses=self.losses)
        reg_loss = tfd.kl_divergence(z_enc, z_plan)
        loss = act_loss + self.beta * reg_loss

        # Update metrics (includes the metric that tracks the loss)
        self.total_loss_tracker.update_state(loss)
        self.action_loss_tracker.update_state(act_loss)
        self.reg_loss_tracker.update_state(reg_loss)
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.action_loss_tracker,
            self.reg_loss_tracker
        ]