import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import numpy as np
import os
import re
import string
import random


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


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads, embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

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
    
    
class PositionEmbedding(layers.Layer):
    def __init__(self, maxlen, embed_dim):
        super(PositionEmbedding, self).__init__()
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-2]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        return x + positions
    
    
def create_conditional_transformer(vocab_size, max_len, goal_dim, embed_dim, num_heads, feed_forward_dim=256, num_layers=1):
    goal = layers.Input(shape=(1, goal_dim,), dtype=tf.float32) # so that we can concat easily
    seq = layers.Input(shape=(max_len,), dtype=tf.int32)
    
    goal_embed = layers.Dense(embed_dim, activation='relu', name='goal_embed')(goal) # convert the goal to the same embedding dim as the seq
    token_embeddings = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)(seq) # embed the seq
    x = layers.Concatenate(axis=-2)([goal_embed, token_embeddings])
    
    # 
    embedding_layer = PositionEmbedding(max_len+1, embed_dim)
    x = embedding_layer(x)
    
    for i in range(num_layers):
        x = TransformerBlock(embed_dim, num_heads, feed_forward_dim)(x)
        
    outputs = layers.Dense(vocab_size)(x)
    model = keras.Model(inputs=[goal, seq], outputs={'logits': outputs, 'x':x})
    return model




class conditional_transformer(tf.keras.Model):
    # TODO: Make height width dependent
    def __init__(self, vocab_size, max_len, goal_dim, embed_dim, num_heads, feed_forward_dim=256, num_layers=1):
        super(conditional_transformer, self).__init__()
        self.goal_embed = layers.Dense(embed_dim, activation='relu', name='goal_embed')
        self.token_embeddings = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.embedding_layer = PositionEmbedding(max_len+1, embed_dim)
        

        self.tformer_layers = [TransformerBlock(embed_dim, num_heads, feed_forward_dim) for i in range(num_layers)]
        self.outputs = layers.Dense(vocab_size)
        
       
    def call(self, inputs):
        goal, seq = inputs # seq should be 1, T (indices)
        
        if len(goal.shape) == 2:
            goal = goal[:, tf.newaxis, :] # insert a time dim
        elif len(goal.shape) == 1:
            goal = goal[tf.newaxis, tf.newaxis, :]
            
        goal_embed = self.goal_embed(goal)
        
        if seq is not None:
            seq_embed = self.token_embeddings(seq)
            x = layers.Concatenate(axis=-2)([goal_embed, seq_embed])
        else:
            x = goal_embed
            
        x = self.embedding_layer(x)
        
        
        for l in self.tformer_layers:
            x = l(x)
        
        logits = self.outputs(x)
        
        
        return {'logits': logits, 'x': x}












