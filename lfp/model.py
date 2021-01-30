import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU, Input, LSTM, Concatenate, Masking, Reshape, Lambda, \
    Bidirectional, GRU, LayerNormalization, Bidirectional
from tensorflow.keras.regularizers import l1, l2
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
tfpl = tfp.layers
from lfp.custom_layers import LearnedInitLSTM, LearnedInitGRU


def latent_normal(inputs):
    mu, scale = inputs
    dist = tfd.Normal(loc=mu, scale=scale)
    return dist


def logistic_mixture(inputs, qbits=None):
    """

    :param inputs:
    :param qbits: number of quantisation bits, total quantisation intervals = 2 ** qbits
    :return:
    """
    def logistic_mixture_lambda():
        weightings, mu, scale = inputs
        if qbits is not None:
            dist = tfd.Logistic(loc=mu, scale=scale)
            dist = tfd.QuantizedDistribution(
                distribution=tfd.TransformedDistribution(
                    distribution=dist,
                    bijector=tfb.Shift(shift=-0.5)),
                low=-2 ** qbits / 2.,
                high=2 ** qbits / 2.,
            )
        else:
            dist = tfd.Logistic(loc=mu, scale=scale)

        mixture_dist = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=weightings),
            components_distribution=dist,
            validate_args=True
        )

        if qbits is not None:
            action_limits = tf.constant([1.5, 1.5, 2.2, 3.2, 3.2, 3.2, 1.1])
            mixture_dist = tfd.TransformedDistribution(
                distribution=mixture_dist,
                bijector=tfb.Scale(scale=action_limits / (2 ** qbits / 2.)) # scale to action limits
            )

        return mixture_dist
    return logistic_mixture_lambda


def create_actor(obs_dim, act_dim, goal_dim,
                 layer_size=1024, latent_dim=256, epsilon=1e-4, num_distribs=None, qbits=None, gcbc=False,
                 training=True, return_state=False, **kwargs):
    # params #
    batch_size = None if training else 1
    stateful = not training

    # Input #
    o = Input(shape=(None, obs_dim), batch_size=batch_size, dtype=tf.float32, name='input_obs')
    z = Input(shape=(None, latent_dim), batch_size=batch_size, dtype=tf.float32, name='input_latent')
    g = Input(shape=(None, goal_dim), batch_size=batch_size, dtype=tf.float32, name='input_goals')

    # RNN #
    if gcbc:
        x = Concatenate(axis=-1)([o, g])
    else:
        x = Concatenate(axis=-1)([o, z, g])

    x = Masking(mask_value=0.)(x)
    if return_state:
        x, _, state1 = LSTM(layer_size, return_sequences=True, stateful=stateful, name='LSTM_in_1',
                            return_state=return_state)(x)
        x, _, state2 = LSTM(layer_size, return_sequences=True, stateful=stateful, name='LSTM_in_2',
                            return_state=return_state)(x)
    else:
        x = LSTM(layer_size, return_sequences=True, stateful=stateful, name='LSTM_in_1', return_state=return_state)(x)
        x = LSTM(layer_size, return_sequences=True, stateful=stateful, name='LSTM_in_2', return_state=return_state)(x)

    # Probabilistic Mixture Model #
    if num_distribs is not None:
        weightings = Dense(act_dim * num_distribs, activation=None, name='alpha')(x)
        mu = Dense(act_dim * num_distribs, activation=None, name='mu')(x)
        scale = Dense(act_dim * num_distribs, activation="softplus", name='sigma')(x + epsilon)

        weightings = Reshape((-1, act_dim, num_distribs))(weightings)
        mu = Reshape((-1, act_dim, num_distribs))(mu)
        scale = Reshape((-1, act_dim, num_distribs))(scale)

        actions = tfpl.DistributionLambda(logistic_mixture(qbits=qbits), name='logistic_mix')([weightings, mu, scale])
    else:
        actions = Dense(act_dim, activation=None, name='acts')(x)

    if return_state:
        if gcbc:
            return Model([o, g], [actions, state1, state2])
        else:
            return Model([o, z, g], [actions, state1, state2])
    else:
        if gcbc:
            return Model([o, g], actions)
        else:
            return Model([o, z, g], actions)


def create_encoder(obs_dim, act_dim,
                   layer_size=2048, latent_dim=256, epsilon=1e-4, training=True, **kwargs):
    # Input #
    obs = Input(shape=(None, obs_dim), dtype=tf.float32, name='obs')
    acts = Input(shape=(None, act_dim), dtype=tf.float32, name='acts')

    # Layers #
    x = Concatenate(axis=-1)([obs, acts])
    x = Masking(mask_value=0.)(x)
    x = Bidirectional(LSTM(layer_size // 4, return_sequences=True), merge_mode='concat')(x)
    x = Bidirectional(LSTM(layer_size // 4, return_sequences=False), merge_mode='concat')(x)

    # Latent Variable #
    mu = Dense(latent_dim, activation=None, name='mu')(x)
    scale = Dense(latent_dim, activation="softplus", name='sigma')(x + epsilon)

    mixture = tfpl.DistributionLambda(latent_normal, name='latent_variable')((mu, scale))
    return Model([obs, acts], mixture)


def create_planner(obs_dim, goal_dim,
                   layer_size=2048, latent_dim=256, epsilon=1e-4, training=True, **kwargs):
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
    x = Dense(layer_size // 4, activation="relu", name='layer_1')(x)
    x = Dense(layer_size // 4, activation="relu", name='layer_2')(x)
    x = Dense(layer_size // 4, activation="relu", name='layer_3')(x)
    x = Dense(layer_size // 4, activation="relu", name='layer_4')(x)

    # Latent Variable #
    mu = Dense(latent_dim, activation=None, name='mu')(x)
    scale = Dense(latent_dim, activation="softplus", name='sigma')(x + epsilon)

    mixture = tfpl.DistributionLambda(latent_normal, name='latent_variable')((mu, scale))
    return Model([o_i, o_g], mixture)

# InfoVAE related
def compute_kernel(x, y):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
    return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)