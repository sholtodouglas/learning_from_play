import os
import numpy as np
import tensorflow as tf
import gym
import pandaRL


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


def load_env(JOINTS = False, QUAT=False, RELATIVE=False, arm='UR5'):

    if arm == 'UR5':
        if JOINTS and RELATIVE:
            env = gym.make('UR5PlayRelJoints1Obj-v0')
        elif JOINTS and not RELATIVE:
            env = gym.make('UR5PlayAbsJoints1Obj-v0')
            #env = gym.make('pandaPlayJoints-v0')
        elif not JOINTS and RELATIVE and QUAT:
            env = gym.make('UR5PlayRel1Obj-v0')
        elif not JOINTS and RELATIVE and not QUAT:
            env = gym.make('UR5PlayRelRPY1Obj-v0')
        elif not JOINTS and not RELATIVE and not QUAT:
            env = gym.make('UR5PlayAbsRPY1Obj-v0')
        else:
            env = gym.make('UR5Play1Obj-v0')
    else:
        if JOINTS and RELATIVE:
            env = gym.make('pandaPlayRelJoints1Obj-v0')
        elif JOINTS and not RELATIVE:
            env = gym.make('pandaPlayAbsJoints1Obj-v0')
            #env = gym.make('pandaPlayJoints-v0')
        elif not JOINTS and RELATIVE and QUAT:
            env = gym.make('pandaPlayRel1Obj-v0')
        elif not JOINTS and RELATIVE and not QUAT:
            env = gym.make('pandaPlayRelRPY1Obj-v0')
        elif not JOINTS and not RELATIVE and not QUAT:
            env = gym.make('pandaPlayAbsRPY1Obj-v0')
        else:
            env = gym.make('pandaPlay1Obj-v0')


    return env
        