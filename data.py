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


# IO function for reading pickle files - we return as the original dictionary
def load_data(filename):
    with open(filename.numpy(), 'rb') as f:
        traj_dict = pickle.load(f)
    return traj_dict['observations'], traj_dict['actions']

# Probably don't need to preprocess for this data
def prepro_data(trajectory):
    pass

def make_tf_dataset(base_path='../relay-policy-learning', batch_size=1, prefetch_size=None, n_threads=1):
    filenames = base_path+"/kitchen_demos_multitask/*/*.pkl"

    with tf.device('/cpu:0'):
        dataset = tf.data.Dataset.list_files(filenames)
        #     dataset = dataset.shuffle(len(filenames))
        dataset = dataset.map(lambda filename: tf.py_function(load_data, [filename], [tf.float32, tf.float32]),
                              num_parallel_calls=n_threads)
        #     dataset = dataset.map(prepro_data, num_parallel_calls=4)
        dataset = dataset.batch(batch_size, drop_remainder=True)  # do we want fixed or variable length sequences
        #     dataset = dataset.window(30, shift=30, drop_remainder=True)
        dataset = dataset.prefetch(prefetch_size)

    return dataset



class RobotSeqDataset():
    def __init__(self, dataset, batch_size=64, seq_len=40, overlap=1.0, prefetch_size=None, train_test_split=0.8, seed=42):
        self.N_TRAJS = len(dataset)

        # Split into train and validation datasets
        # List of trajectory dicts
        self._train_data = dataset[:int(self.N_TRAJS*train_test_split)] # raw data - private
        self._valid_data = dataset[int(self.N_TRAJS*train_test_split):]
        self.train_data = []
        self.valid_data = []

        # Use the obs indices to get the full state of the env, both for us as obs, and for resetting.
        self.START_OBS_IDX, self.END_OBS_IDX = 0,30
        # Get just the dimensions of the goal for appending to the state
        self.START_GOAL_IDX, self.END_GOAL_IDX = 9,30

        self.BATCH_SIZE = batch_size
        self.PREFETCH_SIZE = prefetch_size
        self.OVERLAP = overlap

        self.SEQ_LEN = seq_len
        self.OBS_DIM = 30
        self.ACT_DIM = 9

        self.random_obj = random.Random(seed)

    def create_goal(self, trajectory, ti, tf):
        return np.tile(trajectory['observations'][tf, self.START_GOAL_IDX:self.END_GOAL_IDX], (tf-ti,1))

    def traj_to_subtrajs(self, trajectory):
        """
        Converts a T-length trajectory into M subtrajectories of length SEQ_LEN, pads time dim to SEQ_LEN
        """
        T = len(trajectory['observations'])
        subtrajs = []
        window_size = int(self.SEQ_LEN*self.OVERLAP)
        for ti in range(0,T,window_size):
            tf = min(ti + self.SEQ_LEN, T-1) # Truncate subtrajs at the end of the trajectory
            pad_len = self.SEQ_LEN-(tf-ti)
            time_padding = ((0,pad_len),(0,0))
            subtraj_dict = {
                           'observations':np.pad(trajectory['observations'][ti:tf,self.START_OBS_IDX:self.END_OBS_IDX], time_padding)
                            , 'actions':np.pad(trajectory['actions'][ti:tf], time_padding)
                            , 'goals':np.pad(self.create_goal(trajectory, ti, tf), time_padding)
                            , 'loss_mask': np.pad(np.ones(tf-ti), time_padding[0])
                            , 'init_qpos':None # Just use obs
                            , 'init_qvel':None # Todo: figure this out later - apparently velocities not used
                            }
            subtrajs.append(subtraj_dict)
        return subtrajs

    def convert_dataset(self):
        """ Converts raw dataset to a shuffled subtraj dataset """
        for train_sample in self._train_data:
            self.train_data.extend(self.traj_to_subtrajs(train_sample))

        for valid_sample in self._valid_data:
            self.valid_data.extend(self.traj_to_subtrajs(valid_sample))

    def create_tf_ds(self, ds_type='train'):
        dataset = self.train_data if ds_type=='train' else self.valid_data
        def gen():
            for d in dataset:
                yield (d['observations'], d['actions'], d['goals'], d['loss_mask'])

        with tf.device('/cpu:0'):
            tf_ds =  tf.data.Dataset.from_generator(
                        gen
                        , output_types = (tf.float32, tf.float32, tf.float32, tf.float32)
                        , output_shapes = ((None,self.OBS_DIM), (None,self.ACT_DIM), (None,self.OBS_DIM-self.ACT_DIM), (None)
            ))
            tf_ds = tf_ds.shuffle(len(dataset))
            tf_ds = tf_ds.batch(self.BATCH_SIZE, drop_remainder=True)
            tf_ds = tf_ds.prefetch(self.PREFETCH_SIZE)
        return tf_ds
    
    
    
    
############## Everything from here for pybullet style #################################
def load_data(path, keys):
    cnt = Counter()
    dataset = []
    for demo in os.listdir(path):
        
        traj = np.load(path+demo+'/data.npz')
        traj = {key:traj[key] for key in keys}
        reset_states = []
        for i in range(0, len(traj[keys[0]])):
            # these are needed for deterministic resetting
            reset_states.append(path+demo+'/env_states/'+str(i)+'.bullet')
        traj['reset_states'] = reset_states
        traj['reset_idx'] = int(demo)
        dataset.append(traj)
        
        cnt[len(traj[keys[0]])]+=1
        
    return dataset,cnt