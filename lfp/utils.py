import os
import numpy as np
import tensorflow as tf



def load_weights(path, actor, encoder=None, planner=None, cnn=None, step=""):
    '''
    Load weights function distinct from the trainer to make deploy operation more lightweight
    '''
    if 'checkpoint' in os.listdir(path):
        # Then it was saved using checkpoints
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), actor=actor, encoder=encoder, planner=planner)
        if cnn is not None: # Also load cnn if that is 
            ckpt = tf.train.Checkpoint(step=tf.Variable(1), actor=actor, encoder=encoder, planner=planner, cnn=cnn)
        ckpt.restore(tf.train.latest_checkpoint(path)).expect_partial()
        print('Checkpoint restored')
    else:
        actor.load_weights(f'{path}/model' + step + '.h5')
        if planner is not None: planner.load_weights(f'{path}/planner' + step + '.h5')
        if encoder is not None: encoder.load_weights(f'{path}/encoder' + step + '.h5')
        if cnn is not None: cnn.load_weights(f'{path}/cnn'+step+'.h5')

