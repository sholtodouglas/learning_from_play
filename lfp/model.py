import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU, Input, LSTM, Concatenate, Masking, Reshape, Lambda, \
    Bidirectional, GRU, LayerNormalization, Bidirectional, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.regularizers import l1, l2
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
tfpl = tfp.layers
from lfp.custom_layers import LearnedInitLSTM, LearnedInitGRU
from tensorflow.keras.layers.experimental.preprocessing import Rescaling


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

        actions = tfpl.DistributionLambda(logistic_mixture, name='logistic_mix')([weightings, mu, scale], qbits)
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
    x = Bidirectional(LSTM(layer_size, return_sequences=True), merge_mode='concat')(x)
    x = Bidirectional(LSTM(layer_size, return_sequences=False), merge_mode='concat')(x)

    # Latent Variable #
    mu = Dense(latent_dim, activation=None, name='mu')(x)
    scale = Dense(latent_dim, activation="softplus", name='sigma')(x + epsilon)

    mixture = tfpl.DistributionLambda(latent_normal, name='latent_variable')((mu, scale))
    return Model([obs, acts], mixture)


def create_discrete_encoder(obs_dim, act_dim, layer_size=2048, latent_dim=1024, **kwargs):
    # Input #
    obs = Input(shape=(None, obs_dim), dtype=tf.float32, name='obs')
    acts = Input(shape=(None, act_dim), dtype=tf.float32, name='acts')

    # Layers #
    x = Concatenate(axis=-1)([obs, acts])
    x = Masking(mask_value=0.)(x)
    x = Bidirectional(LSTM(layer_size, return_sequences=True), merge_mode='concat')(x)
    x = Bidirectional(LSTM(layer_size, return_sequences=False), merge_mode='concat')(x)

    logits = Dense(latent_dim, name='to_vocab')(x)
    return Model([obs, acts], logits)


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
    x = Dense(layer_size, activation="relu", name='layer_1')(x)
    x = Dense(layer_size, activation="relu", name='layer_2')(x)
    x = Dense(layer_size, activation="relu", name='layer_3')(x)
    x = Dense(layer_size, activation="relu", name='layer_4')(x)

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



# Standard CNN boi
def create_vision_network(img_height, img_width, embedding_size = 256):

  return Sequential([
  Rescaling(1./255, input_shape=(img_height, img_width, 3)), # put it here for portability
  Conv2D(32, 3, padding='same', activation='relu'),
  MaxPooling2D(),
  Conv2D(32, 3, padding='same', activation='relu'),
  MaxPooling2D(),
  Conv2D(64, 3, padding='same', activation='relu'),
  MaxPooling2D(),
  Conv2D(64, 3, padding='same', activation='relu'),
  MaxPooling2D(),
  Conv2D(128, 3, padding='same', activation='relu'),
  MaxPooling2D(),
  Conv2D(128, 3, padding='same', activation='relu'),
  MaxPooling2D(),
  Conv2D(64, 3, padding='same', activation='relu', name='features'),
  Flatten(),
  Dense(512, activation='relu'),
  Dense(embedding_size),  
], name = 'feature_encoder')

# Has a cheeky 10M params but ok. This is the option which uses spatial softmax. 
class cnn(tf.keras.Model):
    # TODO: Make height width dependent
    def __init__(self,  img_height=200, img_width = 200, img_channels=3, embedding_size=64):
        super(cnn, self).__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.img_channels = img_channels
        self.rescaling = Rescaling(1./255, input_shape=(img_height, img_width, img_channels)) # put it here for portability
        self.conv1 = Conv2D(32, 8, strides=(4,4), padding='same', activation='relu', name='c1')
        self.conv2 = Conv2D(64, 4, strides=(2,2), padding='same', activation='relu', name='c2')
        self.conv3 = Conv2D(64, 4, strides=(2,2), padding='same', activation='relu', name='c3')
        self.conv4 = Conv2D(64, 3, strides=(1,1), padding='same', activation='relu', name='c4')
        # In between these, do a spatial softmax
        self.flatten = Flatten()
        self.dense1 = Dense(512, activation='relu')
        self.dense2 = Dense(embedding_size)
        
    def call(self, inputs):
        x = self.rescaling(inputs)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        pre_softmax = self.conv4(x)
        
        # Assume features is of size [N, H, W, C] (batch_size, height, width, channels).
        # Transpose it to [N, C, H, W], then reshape to [N * C, H * W] to compute softmax
        # jointly over the image dimensions. 
        N, H, W, C = pre_softmax.shape
        pre_softmax = tf.reshape(tf.transpose(pre_softmax, [0, 3, 1, 2]), [N * C, H * W])
        softmax = tf.nn.softmax(pre_softmax)
        # Reshape and transpose back to original format.
        softmax = tf.transpose(tf.reshape(softmax, [N, C, H, W]), [0, 2, 3, 1])
        x = self.flatten(softmax)
        x = self.dense1(x)
        return self.dense2(x)
    
