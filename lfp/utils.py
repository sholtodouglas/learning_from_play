import os
import numpy as np
import tensorflow as tf
import gym

def load_weights(path, actor, encoder=None, planner=None, cnn=None,gripper_cnn=None, step=""):
    '''
    Load weights function distinct from the trainer to make deploy operation more lightweight
    '''
    if 'checkpoint' in os.listdir(path):
        # Then it was saved using checkpoints
        
        
        if gripper_cnn is not None and cnn is not None:
            ckpt = tf.train.Checkpoint(step=tf.Variable(1), actor=actor, encoder=encoder, planner=planner, cnn=cnn, gripper_cnn=gripper_cnn)
        elif cnn is not None: # Also load cnn if that is 
            ckpt = tf.train.Checkpoint(step=tf.Variable(1), actor=actor, encoder=encoder, planner=planner, cnn=cnn)
        else:
            ckpt = tf.train.Checkpoint(step=tf.Variable(1), actor=actor, encoder=encoder, planner=planner)
            
        ckpt.restore(tf.train.latest_checkpoint(path)).expect_partial()
        print('Checkpoint restored')
    else:
        actor.load_weights(f'{path}/model' + step + '.h5')
        if planner is not None: planner.load_weights(f'{path}/planner' + step + '.h5')
        if encoder is not None: encoder.load_weights(f'{path}/encoder' + step + '.h5')
        if cnn is not None: cnn.load_weights(f'{path}/cnn'+step+'.h5')


def build_cnn(cnn):
    x = tf.zeros((1, cnn.img_height, cnn.img_width, cnn.img_channels))
    cnn(x)

def images_to_2D_features(imgs, proprioceptive_features, goal_imgs, cnn):
    '''
    This function is used in training and plotting, so just deduplicating it here.
    SImply - takes the imgs + propriocepetive features and does the necessary reshaping + running of cnn
    '''
    imgs, proprioceptive_features, goal_imgs = imgs, proprioceptive_features, goal_imgs
    B, T, H, W, C = imgs.shape
    imgs = tf.reshape(imgs, [B * T, H, W, C])
    img_embeddings = tf.reshape(cnn(imgs), [B, T, -1])
    states = tf.concat([img_embeddings, proprioceptive_features],-1)  # gets both the image and it's own xyz ori and angle as pose
    if len(goal_imgs.shape) == 5:
        goal_imgs = tf.reshape(goal_imgs, [B * T, H, W, C])
        goals = tf.reshape(cnn(goal_imgs), [B, T, -1])
    else: # It came in without a time dimension as we are just sending it to the planner - as it does in the plotting code
        goal_imgs = tf.reshape(goal_imgs, [B, H, W, C])
        goals = tf.reshape(cnn(goal_imgs), [B, -1])
    return states, goals


def load_env(JOINTS = False, QUAT=False, RELATIVE=False, arm='UR5'):
    import roboticsPlayroomPybullet

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
        


class LocalTPUClusterResolver(
    tf.distribute.cluster_resolver.TPUClusterResolver):
    def __init__(self):
        self._tpu = ''
        self.task_type = 'worker'
        self.task_id = 0

    def master(self, task_type=None, task_id=None, rpc_layer=None):
        return None

    def cluster_spec(self):
        return tf.train.ClusterSpec({})

    def get_tpu_system_metadata(self):
        return tf.tpu.experimental.TPUSystemMetadata(
            num_cores=8,
            num_hosts=1,
            num_of_cores_per_host=8,
            topology=None,
            devices=tf.config.list_logical_devices())

    def num_accelerators(self,
                        task_type=None,
                        task_id=None,
                        config_proto=None):
        return {'TPU': 8}
