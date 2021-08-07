import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU, Input, LSTM, Concatenate, Masking, Reshape, Lambda, \
    Bidirectional, GRU, LayerNormalization, Bidirectional, Conv2D, Conv1D, MaxPooling2D, Flatten, LayerNormalization, Layer, Embedding, MultiHeadAttention, Dropout
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
                 training=True, return_state=False, discrete=False, disc_embed_size=64, **kwargs):
    # params #
    batch_size = None if training else 1
    stateful = not training

    # Input #
    o = Input(shape=(None, obs_dim), batch_size=batch_size, dtype=tf.float32, name='input_obs')
    z = Input(shape=(None, latent_dim), batch_size=batch_size, dtype=tf.float32, name='input_latent')
    g = Input(shape=(None, goal_dim), batch_size=batch_size, dtype=tf.float32, name='input_goals')

    # RNN #
    if discrete:
        embed = Dense(disc_embed_size, activation = 'relu', name='disc_to_cts_embedding')(z)
        x = Concatenate(axis=-1)([o, embed, g])
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


def create_encoder(enc_in_dim,
                   layer_size=2048, latent_dim=256, epsilon=1e-4, training=True, **kwargs):
    # Input #
    inputs = Input(shape=(None, enc_in_dim), dtype=tf.float32, name='encoder_in')

    # Layers #
    x = Masking(mask_value=0.)(inputs)
    x = Bidirectional(LSTM(layer_size, return_sequences=True), merge_mode='concat')(x)
    x = Bidirectional(LSTM(layer_size, return_sequences=False), merge_mode='concat')(x)

    # Latent Variable #
    mu = Dense(latent_dim, activation=None, name='mu')(x)
    scale = Dense(latent_dim, activation="softplus", name='sigma')(x + epsilon)

    mixture = tfpl.DistributionLambda(latent_normal, name='latent_variable')((mu, scale))
    return Model([inputs], mixture)


# def create_discrete_encoder(enc_in_dim, layer_size=2048, latent_dim=1024, **kwargs):
#     # Input #
#     inputs = Input(shape=(None, enc_in_dim), dtype=tf.float32, name='encoder_in')

#     # Layers #
#     x = Masking(mask_value=0.)(inputs)
#     x = Bidirectional(LSTM(layer_size, return_sequences=True), merge_mode='concat')(x)
#     x = Bidirectional(LSTM(layer_size, return_sequences=False), merge_mode='concat')(x)

#     logits = Dense(latent_dim, name='to_vocab')(x)
#     return Model([inputs], logits)

def create_discrete_encoder(enc_in_dim, layer_size=128, latent_dim=64, reductions=3, **kwargs):
    # Input #
    inputs = Input(shape=(None, enc_in_dim), dtype=tf.float32, name='encoder_in')

    # Layers #
    x = Masking(mask_value=0.)(inputs)
    x = Bidirectional(LSTM(layer_size, return_sequences=True), merge_mode='concat')(x)
    x = Bidirectional(LSTM(layer_size, return_sequences=True), merge_mode='concat')(x)
    for l in range(reductions-1):
        print(l)
        x = Conv1D(layer_size, kernel_size=3, strides=2, padding="same")(x)

    embed = Conv1D(latent_dim, kernel_size=3, strides=2, padding="same")(x)
    return Model([inputs], embed)


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

def create_discrete_planner(obs_dim, goal_dim,
                   layer_size=2048, latent_dim=256, epsilon=1e-4, training=True, **kwargs):
    '''
    takes in size B, N_TILES, D for start_state and goal_state
    LSTM then predicts which discrete plan it should be for each tile
    '''
    # params #
    batch_size = None if training else 1
    stateful = not training

    # Input #
    o = Input(shape=(None, obs_dim), batch_size=batch_size, dtype=tf.float32, name='input_obs')
    g = Input(shape=(None, goal_dim), batch_size=batch_size, dtype=tf.float32, name='input_goals')

    # RNN #
    x = Concatenate(axis=-1)([o, g])
    x = LSTM(layer_size, return_sequences=True, stateful=stateful, name='LSTM_in_1', return_state=False)(x)
    x = LSTM(layer_size, return_sequences=True, stateful=stateful, name='LSTM_in_2', return_state=False)(x)
    tokens = Dense(latent_dim, name='acts')(x)



    return Model([o, g], tokens) 

# maps from sentence embedding space to goal dim space
def create_goal_space_mapper(input_embedding_dim, goal_embedding_dim,
                   layer_size=2048, **kwargs):
    # params #
    batch_size = None

    # Input #
    input_embeddings = Input(shape=(input_embedding_dim,), batch_size=batch_size, dtype=tf.float32,
                name='lang_embeds')  # embeddings created by MUSE or equiv

    # Layers #
    x = Masking(mask_value=0.)(input_embeddings)
    x = Dense(layer_size, activation="relu", name='layer_1')(x)
    x = Dense(layer_size, activation="relu", name='layer_2')(x)

    goal_embeddings = Dense(goal_embedding_dim, activation=None, name='goal_space')(x)

    return Model(input_embeddings, goal_embeddings)

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



# # Standard CNN boi
# def create_vision_network(img_height, img_width, embedding_size = 256):

#   return Sequential([
#   Rescaling(1./255, input_shape=(img_height, img_width, 3)), # put it here for portability
#   Conv2D(32, 3, padding='same', activation='relu'),
#   MaxPooling2D(),
#   Conv2D(32, 3, padding='same', activation='relu'),
#   MaxPooling2D(),
#   Conv2D(64, 3, padding='same', activation='relu'),
#   MaxPooling2D(),
#   Conv2D(64, 3, padding='same', activation='relu'),
#   MaxPooling2D(),
#   Conv2D(128, 3, padding='same', activation='relu'),
#   MaxPooling2D(),
#   Conv2D(128, 3, padding='same', activation='relu'),
#   MaxPooling2D(),
#   Conv2D(64, 3, padding='same', activation='relu', name='features'),
#   Flatten(),
#   Dense(512, activation='relu'),
#   Dense(embedding_size),  
# ], name = 'feature_encoder')

# # Has a cheeky 10M params but ok. This is the option which uses spatial softmax. 
# class cnn(tf.keras.Model):
#     # TODO: Make height width dependent
#     def __init__(self,  img_height=128, img_width = 128, img_channels=3, embedding_size=64):
#         super(cnn, self).__init__()
#         self.img_height = img_height
#         self.img_width = img_width
#         self.img_channels = img_channels
#         self.rescaling = Rescaling(1./255, input_shape=(img_height, img_width, img_channels)) # put it here for portability
#         self.conv1 = Conv2D(32, 8, strides=(4,4), padding='same', activation='relu', name='c1')
#         self.conv2 = Conv2D(64, 4, strides=(2,2), padding='same', activation='relu', name='c2')
#         self.conv3 = Conv2D(64, 4, strides=(2,2), padding='same', activation='relu', name='c3')
#         self.conv4 = Conv2D(64, 3, strides=(1,1), padding='same', activation='relu', name='c4')
#         # In between these, do a spatial softmax
#         self.flatten = Flatten()
#         self.dense1 = Dense(512, activation='relu')
#         self.dense2 = Dense(embedding_size)
        
#     def call(self, inputs):
#         x = self.rescaling(inputs)
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         pre_softmax = self.conv4(x)
        
#         # Assume features is of size [N, H, W, C] (batch_size, height, width, channels).
#         # Transpose it to [N, C, H, W], then reshape to [N * C, H * W] to compute softmax
#         # jointly over the image dimensions. 
#         N, H, W, C = pre_softmax.shape
#         pre_softmax = tf.reshape(tf.transpose(pre_softmax, [0, 3, 1, 2]), [N * C, H * W])
#         softmax = tf.nn.softmax(pre_softmax)
#         # Reshape and transpose back to original format.
#         softmax = tf.transpose(tf.reshape(softmax, [N, C, H, W]), [0, 2, 3, 1])
#         x = self.flatten(softmax)
#         x = self.dense1(x)
#         return self.dense2(x)
    

# Has a cheeky 10M params but ok. This is the option which uses spatial softmax. 




class spatial_softmax_cnn(tf.keras.Model):
    # TODO: Make height width dependent
    def __init__(self,  img_height=128, img_width = 128, img_channels=3, embedding_size=64, return_spatial_softmax = False):
        super(spatial_softmax_cnn, self).__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.img_channels = img_channels
        self.rescaling = Rescaling(1./255, input_shape=(img_height, img_width, img_channels)) # put it here for portability
        self.conv1 = Conv2D(32, 8, strides=(4,4), padding='same', activation='relu', name='c1')
        self.conv2 = Conv2D(64, 4, strides=(2,2), padding='same', activation='relu', name='c2')
        self.conv3 = Conv2D(64, 3, strides=(1,1), padding='same', activation='relu', name='c3')
        # In between these, do a spatial softmax
        self.flatten = Flatten()
        self.dense1 = Dense(512, activation='relu')
        self.dense2 = Dense(embedding_size)
        self.return_spatial_softmax = return_spatial_softmax

         
        
    def call(self, inputs):
        x = self.rescaling(inputs)
        x = self.conv1(x)
        x = self.conv2(x)
        pre_softmax = self.conv3(x)
        # pre_softmax = self.conv4(x)
        
        # Assume features is of size [N, H, W, C] (batch_size, height, width, channels).
        # Transpose it to [N, C, H, W], then reshape to [N * C, H * W] to compute softmax
        # jointly over the image dimensions. 
        N, H, W, C = pre_softmax.shape
        pre_softmax = tf.reshape(tf.transpose(pre_softmax, [0, 3, 1, 2]), [N * C, H * W])
        softmax = tf.nn.softmax(pre_softmax)
        # Reshape and transpose back to original format.
        softmax = tf.transpose(tf.reshape(softmax, [N, C, H, W]), [0, 2, 3, 1]) # N, H, W, C

        # Expand dims by 1
        softmax  = tf.expand_dims(softmax, -1)

        x, y = tf.range(0, W)/W, tf.range(0, H)/H # so that feature locations are on a 0-1 scale not 0-128
        X,Y = tf.meshgrid(x,y)
        # Image coords is a tensor of size [H,W,2] representing the image coordinates of each pixel
        image_coords = tf.cast(tf.stack([X,Y],-1), tf.float32)
        image_coords= tf.expand_dims(image_coords, 2)
        # multiply to get feature locations
        spatial_soft_argmax = tf.reduce_sum(softmax * image_coords, axis=[1,2])
            
        x = self.flatten(spatial_soft_argmax)
        x = self.dense1(x)
        
        return self.dense2(x), spatial_soft_argmax
        

class intensities_spatial_softmax_cnn(tf.keras.Model):
    # TODO: Make height width dependent
    def __init__(self,  img_height=128, img_width = 128, img_channels=3, embedding_size=64, return_spatial_softmax = False):
        super(intensities_spatial_softmax_cnn, self).__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.img_channels = img_channels
        self.rescaling = Rescaling(1./255, input_shape=(img_height, img_width, img_channels)) # put it here for portability
        self.conv1 = Conv2D(32, 8, strides=(4,4), padding='same', activation='relu', name='c1')
        self.conv2 = Conv2D(64, 4, strides=(2,2), padding='same', activation='relu', name='c2')
        self.conv3 = Conv2D(64, 3, strides=(1,1), padding='same', activation='relu', name='c3')
        # In between these, do a spatial softmax
        self.flatten = Flatten()
        self.dense1 = Dense(512, activation='relu')
        self.dense2 = Dense(embedding_size)
        self.return_spatial_softmax = return_spatial_softmax

         
        
    def call(self, inputs):
        x = self.rescaling(inputs)
        x = self.conv1(x)
        x = self.conv2(x)
        pre_softmax = self.conv3(x)
        # pre_softmax = self.conv4(x)
        
        # Assume features is of size [N, H, W, C] (batch_size, height, width, channels).
        # Transpose it to [N, C, H, W], then reshape to [N * C, H * W] to compute softmax
        # jointly over the image dimensions. 
        N, H, W, C = pre_softmax.shape
        pre_softmax = tf.reshape(tf.transpose(pre_softmax, [0, 3, 1, 2]), [N * C, H * W])
        softmax = tf.nn.softmax(pre_softmax)
        # Reshape and transpose back to original format.
        softmax = tf.transpose(tf.reshape(softmax, [N, C, H, W]), [0, 2, 3, 1]) # N, H, W, C

        # Expand dims by 1
        softmax  = tf.expand_dims(softmax, -1)

        x, y = tf.range(0, W)/W, tf.range(0, H)/H # so that feature locations are on a 0-1 scale not 0-128
        X,Y = tf.meshgrid(x,y)
        # Image coords is a tensor of size [H,W,2] representing the image coordinates of each pixel
        image_coords = tf.cast(tf.stack([X,Y],-1), tf.float32)
        image_coords= tf.expand_dims(image_coords, 2)
        # multiply to get feature locations
        spatial_soft_argmax = tf.reduce_sum(softmax * image_coords, axis=[1,2])

        # Get indices corresponding to each
        batch_indices =tf.reshape(tf.repeat(tf.range(0,N,1)[tf.newaxis,:], C), [N,C])[:,:,tf.newaxis] # 0,0,0, 1,1,1, etc as batch indices
        keypoint_indices = tf.tile(tf.range(0,C,1)[tf.newaxis, :], [N,1])[:, :, tf.newaxis] # numbers 1,2,3... 1,2,3... keypoints, batches appropriately
        assert W == H # this next step is currently only coded for squares
        keypoint_img_indices = tf.reverse(tf.cast(spatial_soft_argmax * W, tf.int32), [-1]) # gather nd has opposite axes to images, x is y, y is x
        gather_indices = tf.concat([batch_indices, keypoint_img_indices, keypoint_indices], axis = -1)
        feature_intensities = tf.gather_nd(softmax, gather_indices) # N, C, 1
        
        keypoints_with_intensities = tf.concat([feature_intensities, spatial_soft_argmax], -1)
        
        x = self.flatten(keypoints_with_intensities)
            
        x = self.dense1(x)
        
        return self.dense2(x), keypoints_with_intensities





class impala_cnn(tf.keras.Model):
    def __init__(self,  img_height=128, img_width = 128, img_channels=3, embedding_size=64, return_spatial_softmax = False, l1=16, l2=32, l3=32):
        super(impala_cnn, self).__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.img_channels = img_channels
        self.rescaling = Rescaling(1./255, input_shape=(img_height, img_width, img_channels)) # put it here for portability
        self.conv_1 = Conv2D(l1, 3, strides=(2,2), padding='same', activation='relu', name='c1')
        self.res_1_1 =  Conv2D(l1, 3, strides=(1,1), padding='same', activation='relu', name='r1_1')
        self.res_1_2 =  Conv2D(l1, 3, strides=(1,1), padding='same', activation='relu', name='r1_2')

        self.conv_2 = Conv2D(l2, 3, strides=(2,2), padding='same', activation='relu', name='c2')
        self.res_2_1 =  Conv2D(l2, 3, strides=(1,1), padding='same', activation='relu', name='r2_1')
        self.res_2_2 =  Conv2D(l2, 3, strides=(1,1), padding='same', activation='relu', name='r2_2')

        self.conv_3 = Conv2D(l3, 3, strides=(2,2), padding='same', activation='relu', name='c3')
        self.res_3_1 =  Conv2D(l3, 3, strides=(1,1), padding='same', activation='relu', name='r3_1')
        self.res_3_2 =  Conv2D(l3, 3, strides=(1,1), padding='same', activation='relu', name='r3_2')

        # In between these, do a spatial softmax
        self.flatten = Flatten()
        self.dense1 = Dense(256, activation='relu')
        self.dense2 = Dense(embedding_size)
        self.return_spatial_softmax = return_spatial_softmax

         
        
    def call(self, inputs):
        x = self.rescaling(inputs)
        x = self.conv_1(x)
        r1 = self.res_1_1(x)
        x = self.res_1_2(r1) + x

        x = self.conv_2(x)
        r1 = self.res_2_1(x)
        x = self.res_2_2(r1) + x

        x = self.conv_3(x)
        r1 = self.res_3_1(x)
        x = self.res_3_2(r1) + x

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x, 0 # for compat with spatial softmax returns


class deep_impala_cnn(impala_cnn):
    def __init__(self,  img_height=128, img_width = 128, img_channels=3, embedding_size=64, return_spatial_softmax = False):
        super(deep_impala_cnn, self).__init__(img_height, img_width, img_channels, embedding_size, return_spatial_softmax, l1=64, l2=128, l3=128)


CNN_DICT= {'spatial_softmax': spatial_softmax_cnn, 'intensities_spatial_softmax': intensities_spatial_softmax_cnn, 'impala': impala_cnn, 'deep_impala': deep_impala_cnn}






###############################################################################




def causal_attention_mask(batch_size, n_dest, n_src, dtype):
    """
    Mask the upper half of the dot product matrix in self attention.
    This prevents flow of information from future tokens to current token.
    1's in the lower triangle, counting from the lower right corner.
    """
    i = tf.range(n_dest)[:, None]
    j = tf.range(n_src)
    m = i >= j - n_src + n_dest
    mask = tf.cast(m, dtype)
    mask = tf.reshape(mask, [1, n_dest, n_src])
    mult = tf.concat(
        [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
    )
    return tf.tile(mask, mult)


class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads, embed_dim)
        self.ffn = Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        attention_output = self.att(inputs, inputs, attention_mask=causal_mask)
        attention_output = self.dropout1(attention_output)
        out1 = self.layernorm1(inputs + attention_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)
    
    
class PositionEmbedding(Layer):
    def __init__(self, maxlen, embed_dim):
        super(PositionEmbedding, self).__init__()
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-2]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        return x + positions
    
    
# def create_conditional_transformer(vocab_size, max_len, embed_dim, num_heads, feed_forward_dim=256, num_layers=1):
#     goal = Input(shape=(1, goal_dim,), dtype=tf.float32) # so that we can concat easily
#     seq = Input(shape=(max_len,), dtype=tf.int32)
    
#     goal_embed = Dense(embed_dim, activation='relu', name='goal_embed')(goal) # convert the goal to the same embedding dim as the seq
#     token_embeddings = Embedding(input_dim=vocab_size, output_dim=embed_dim)(seq) # embed the seq
#     x = Concatenate(axis=-2)([goal_embed, token_embeddings])
    
#     # 
#     embedding_layer = PositionEmbedding(max_len+1, embed_dim)
#     x = embedding_layer(x)
    
#     for i in range(num_layers):
#         x = TransformerBlock(embed_dim, num_heads, feed_forward_dim)(x)
        
#     outputs = Dense(vocab_size)(x)
#     model = Model(inputs=[goal, seq], outputs={'logits': outputs, 'x':x})
#     return model




class conditional_transformer(Model):
    # TODO: Make height width dependent
    def __init__(self, vocab_size, max_len,embed_dim, num_heads, feed_forward_dim=256, num_layers=1):
        super(conditional_transformer, self).__init__()
        self.goal_embed = Dense(embed_dim, activation='relu', name='goal_embed')
        self.state_embed = Dense(embed_dim, activation='relu', name='state_embed')
        self.token_embeddings = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.embedding_layer = PositionEmbedding(max_len+1, embed_dim)
        

        self.tformer_layers = [TransformerBlock(embed_dim, num_heads, feed_forward_dim) for i in range(num_layers)]
        self.outputs = Dense(vocab_size)

    def expand(self, input):
        if len(input.shape) == 2:
            return input[:, tf.newaxis, :] # insert a time dim
        elif len(input.shape) == 1:
            return input[tf.newaxis, tf.newaxis, :]
        
       
    def call(self, inputs):
        current_state, goal, seq = inputs # seq should be 1, T (indices)
        
        current_state = self.expand(current_state)
        goal = self.expand(goal)
       
        state_embed = self.state_embed(current_state)
        goal_embed = self.goal_embed(goal)
        
        if seq is not None:
            seq_embed = self.token_embeddings(seq)
            x = Concatenate(axis=-2)([goal_embed, state_embed, seq_embed])
        else:
            x = Concatenate(axis=-2)([goal_embed, state_embed])
            
        x = self.embedding_layer(x)
        
        
        for l in self.tformer_layers:
            x = l(x)
        
        logits = self.outputs(x)
        
        
        return {'logits': logits, 'x': x}