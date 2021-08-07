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
import matplotlib.pyplot as plt

def log2(x):
    return tf.math.log(x) / tf.math.log(2.)

def nansum(x):
    return tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), x),
                         axis=-1)

def entropy(p):
    return nansum(-p * log2(p))

class VQ_EMA(tf.keras.Model):
    
    def __init__(self, args, commitment_cost= 1.0, gamma=0.99, epsilon=1e-9):
        self.args = args
        embedding_shape = (args.latent_dim, args.codebook_size)
        input_size = np.product(embedding_shape[:-1])
        max_val = np.sqrt(3 / input_size) * 1.0 # scale 1.0
        self.codebook = tf.Variable(tf.random.uniform(embedding_shape, -max_val, max_val))
        self.gamma = gamma
        self.commitment_cost = commitment_cost
        self.epsilon = epsilon
        
        self.codebook_history = []
        self.commitment_loss_history = []
        self.entropy_history = []
        
    def update_codebook(self, z, encodings):
        cluster_size = tf.reduce_sum(encodings, 0) # N_codebook, the count of how many of the encodings matched to that codebook vector
        
        dw = tf.matmul(tf.transpose(z), encodings) # [_encoding_size, N] x [N, codebook_size] > [E, codebook_size]
        
        normalised_dw = dw / (cluster_size + self.epsilon)# divide by the number of weights there
#         print(dw, cluster_size)
#         print(self.codebook.shape, normalised_dw.shape)
        self.codebook = self.gamma * self.codebook + (1-self.gamma) * normalised_dw # EMA avg
        
    
    def quantise(self, indices):
        '''
        Returns the embeddings from the indices
        '''
        return tf.gather_nd(tf.transpose(self.codebook),tf.transpose(indices[tf.newaxis,:]))
        
    def call(self, z, training=False, record_codebook=False):
        '''
        Takes in a series of z, does the full forward pass
        '''

        flattened_inputs = tf.reshape(z, (-1, self.args.latent_dim))
        d = tf.reduce_sum(flattened_inputs**2, -1)[:, tf.newaxis] -  2*tf.matmul(flattened_inputs , self.codebook)  + tf.reduce_sum(self.codebook**2, 0)[tf.newaxis, :] # [N_latents, N_codebook] - neat trick huh!
        indices = tf.argmax(-d, -1)
        encodings = tf.one_hot(indices, self.args.codebook_size) # N_latents, N_codebook
        
        quantised = self.quantise(indices)
        
        e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantised) - flattened_inputs)**2)
        
        if training:
            self.update_codebook(flattened_inputs, encodings)
        
        commitment_loss = self.commitment_cost * e_latent_loss
        
        # straight through estimator
        quantised = flattened_inputs + tf.stop_gradient(quantised-flattened_inputs)
        
        # logging
        avg_probs = tf.reduce_mean(encodings, 0)
        
        if record_codebook:
            self.codebook_history.append(self.codebook)
            self.commitment_loss_history.append(commitment_loss)
            self.entropy_history.append(entropy(avg_probs))
        
        return {'quantised': quantised, 'commitment_loss':commitment_loss, 'entropy': avg_probs, 'indices': indices}
    
    def plot_codebook_history(self):
        plt.rcParams["figure.figsize"] = (20, 5)
        # TODO UMAP it down to 2D
        plt.subplot(1, 4, 1)
        for c in self.codebook_history:
            plt.scatter(c[0,:], c[1,:], c = ['r', 'g', 'b', 'y'])
        plt.subplot(1, 4, 2)
        plt.plot(np.array(self.codebook_history)[:, 0, :])
        plt.subplot(1, 4, 3)
        plt.plot(self.commitment_loss_history)
        plt.subplot(1, 4, 4)
        plt.plot(self.entropy_history)