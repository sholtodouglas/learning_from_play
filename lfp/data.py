import glob
import pickle
import random
from collections import Counter
import subprocess
import shlex
from multiprocessing.pool import ThreadPool
import tensorflow as tf
import os
import numpy as np
from natsort import natsorted
import sklearn


# Alternatively we could just add other repos to pythonpath
def build_dataset_from_mjl(base_path='../relay-policy-learning', num_cpus=1):
    def call_proc(cmd):
        """ This runs in a separate thread. """
        p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        return (out, err)

    # Probably do 1-2 CPUs - using all CPUs will make you computer completely unusable
    pool = ThreadPool(num_cpus) # multiprocessing.cpu_count()
    results = []
    for path in glob.glob(base_path+'/kitchen_demos_multitask/*'):
        print(path)
        cmd = f'python3 relay-policy-learning/adept_envs/adept_envs/utils/parse_demos.py --env "kitchen_relax-v1" -d "{path}/" -s "40" -v "playback" -r "offscreen"'
        results.append(pool.apply_async(call_proc, (cmd,)))



    # Close the pool and wait for each running task to complete
    pool.close()
    pool.join()
    for result in results:
        out, err = result.get()
        print(f"out: {out.decode()}")
        print(f"err: {err.decode()}")

def create_single_dataset(base_path='../relay-policy-learning'):
    """
    Takes in a filepath a loads all pickle files into one big list
    :return:
    observations - list of observation arrays
    actions - list of action arrays
    cnt - counter object for plotting distribution of seq lens
    """
    traj_dicts = []
    MAX_SEQ_LEN = 0 # overall max of the dataset is 409 - this is overkill
    cnt = Counter()

    filenames = glob.glob(base_path+'/kitchen_demos_multitask/*/*.pkl')
    random.Random(42).shuffle(filenames) # shuffle the order of trajectory files

    for path in filenames:
        with open(path, 'rb') as f:
            traj_dict = pickle.load(f)
        traj_dicts.append(traj_dict)
        cnt[len(traj_dict['observations'])]+=1

    return traj_dicts, cnt

### Play dataset

class PlayDataloader():
    def __init__(self, 
                relative_obs=False,
                relative_act=False,
                quaternion=False,
                joints=False,
                velocity=False,
                normalize=False,
                proprioception=False,

                batch_size=32, # 512*8
                window_size=50,
                min_window_size=20,
                window_shift=1,
                variable_seqs=True, 
                num_workers=4,
                seed=42):
        
        self.relative_obs = relative_obs
        self.relative_act = relative_act
        self.quaternion = quaternion
        self.joints = joints
        self.velocity = velocity
        self.normalize = normalize
        self.proprioception = proprioception
        
        self.batch_size = batch_size
        self.window_size = window_size
        self.min_window_size = min_window_size
        self.window_shift = window_shift
        self.variable_seqs = variable_seqs
        self.shuffle_size = int(batch_size * (window_size / window_shift))
        self.prefetch_size = tf.data.experimental.AUTOTUNE
        self.num_workers = num_workers
        self.seed=seed
        
    @staticmethod
    def print_minutes(dataset):
        dataset_size = dataset['obs'].shape[0]
        secs = dataset_size / 25
        hours = secs // 3600
        minutes = secs // 60 - hours * 60
        print(f"{dataset_size} frames, which is {hours:.0f}hrs {minutes:.0f}m.")
        
    def create_goal_tensor(self, dataset, seq_len=-1):
        ''' Tile final achieved_goal across time dimension '''
        tile_dims = tf.constant([self.window_size, 1], tf.int32)
        goal = tf.tile(dataset['achieved_goals'][seq_len-1,tf.newaxis], tile_dims) # as goal is at an index take seq_len -1
        return goal
        
    def extract(self, paths):
        keys = ['obs','acts','achieved_goals','joint_poses','target_poses','acts_rpy','acts_rpy_rel','velocities','obs_rpy','proprioception']
        dataset = {k:[] for k in keys+['sequence_index','sequence_id']}

        for path in paths:
            obs_act_path = os.path.join(path, 'obs_act_etc/')
            for demo in natsorted(os.listdir(obs_act_path)):
                traj = np.load(obs_act_path+demo+'/data.npz')
                for k in keys:
                    d = traj[k]
                    if len(d.shape) < 2:
                        d = np.expand_dims(d, axis = 1) # was N, should be N,1
                    dataset[k].append(d.astype(np.float32))
                timesteps = len(traj['obs'])
                dataset['sequence_index'].append(np.arange(timesteps, dtype=np.int32).reshape(-1, 1))
                dataset['sequence_id'].append(np.full(timesteps, fill_value=int(demo), dtype=np.int32).reshape(-1, 1))

        # convert to numpy
        for k in keys+['sequence_index','sequence_id']:
            dataset[k] = np.vstack(dataset[k])
            
        self.print_minutes(dataset)
        return dataset
    
    def transform(self, dataset):
        # State representations
        if self.joints:
            obs = tf.concat([dataset['obs'],dataset['joint_poses'][:,:7]], axis=-1)
            gripper = dataset['acts'][:,-1,tf.newaxis]
            acts = tf.concat([dataset['target_poses'], gripper], axis=-1)
            acts_prev = tf.concat([dataset['joint_poses'][:,:6], gripper], axis=-1)
        else: 
            if self.quaternion:
                # TODO: this is confusing me
#                 obs = tf.concat([dataset['obs'],dataset['joint_poses'][:,:7]], axis=-1)
#                 gripper = dataset['acts'][:,-1,tf.newaxis]
#                 acts = tf.concat([dataset['acts'][:,:7], gripper], axis=-1)
#                 acts_prev = tf.concat([dataset['obs'][:,:7], gripper], axis=-1)
                pass
            else: # RPY
                # TODO:
#                 obs = 
#                 gripper = 
#                 acts = 
#                 acts_prev = 
                pass
        
        if self.relative_obs:
            obs = np.diff(obs, axis=0)
        if self.relative_act:
            acts = acts - acts_prev
            
        if self.velocity:
            obs = tf.concat([obs, dataset['velocities']], axis=-1)
        if self.proprioception:
            obs = tf.concat([obs, dataset['proprioception']], axis=-1)
            
        # Variable Seq len
        if self.variable_seqs:
            seq_len = tf.random.uniform(shape=[], minval=self.min_window_size, 
                                        maxval=self.window_size, dtype=tf.int32, seed=self.seed)
            goals = self.create_goal_tensor(dataset, seq_len)
            # Masking
            mask = tf.cast(tf.sequence_mask(seq_len, maxlen=self.window_size), tf.float32) # creates a B*T mask
            multiply_mask = tf.expand_dims(mask, -1)

            obs *= multiply_mask
            goals *= multiply_mask
            acts *= multiply_mask
        else:
            seq_len = self.window_size
            goals = self.create_goal_tensor(dataset, seq_len)
          
        # Key data dimensions
        self.obs_dim = obs.shape[1]
        self.goal_dim = goals.shape[1]
        self.act_dim = acts.shape[1]
        
        # Preprocessing
        if self.normalize:
            sklearn.preprocessing.normalize(obs, norm='l2', axis=1, copy=False)
            sklearn.preprocessing.normalize(goals, norm='l2', axis=1, copy=False)
            sklearn.preprocessing.normalize(acts, norm='l2', axis=1, copy=False)
        
        return {'obs':obs, 
                'acts':acts, 
                'goals':goals, 
                'seq_lens': tf.cast(seq_len, tf.float32), 
                'masks':mask, 
                'dataset_path':dataset['sequence_id'], 
                'tstep_idxs':dataset['sequence_index']}
    
    # TODO: why did we not need this before?? window_lambda is being weird
    @tf.autograph.experimental.do_not_convert   
    def load(self, dataset):
        dataset = tf.data.Dataset.from_tensor_slices(dataset)
        window_lambda = lambda x: tf.data.Dataset.zip(x).batch(self.window_size)
        seq_overlap_filter = lambda x: tf.equal(tf.size(tf.unique(tf.squeeze(x['sequence_id'])).y), 1)
        dataset = dataset\
                .window(size=self.window_size, shift=self.window_shift, stride=1, drop_remainder=True)\
                .flat_map(window_lambda)\
                .filter(seq_overlap_filter)\
                .shuffle(self.shuffle_size)\
                .repeat()\
                .map(self.transform, num_parallel_calls=self.num_workers)\
                .batch(self.batch_size, drop_remainder=True)\
                .prefetch(self.prefetch_size)
        
        self.obs_dim = dataset.element_spec['obs'].shape[-1]
        self.goal_dim = dataset.element_spec['goals'].shape[-1]
        self.act_dim = dataset.element_spec['acts'].shape[-1]
        print(dataset.element_spec)
        return dataset