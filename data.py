import glob
import pickle
import random
from collections import Counter
import subprocess
import shlex
from multiprocessing.pool import ThreadPool
import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE
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

class PyBulletRobotSeqDataset():
    def __init__(self, dataset, batch_size=64, seq_len=32, overlap=1.0, 
                 prefetch_size=AUTOTUNE, train_test_split=0.9, relative_joints=False, 
                 variable_seqs=True, seed=42):
        self.N_TRAJS = len(dataset)

        # Split into train and validation datasets
        # List of trajectory dicts
        if train_test_split == 'last': # just use the last set of demos as validation
            self._train_data = dataset[:-1] #[:-1] # raw data - private
            self._valid_data = dataset[-1:]
        else:
            self._train_data = dataset[:int(self.N_TRAJS*train_test_split)] # raw data - private
            self._valid_data = dataset[int(self.N_TRAJS*train_test_split):]
        self.train_data = []
        self.valid_data = []
        self.BATCH_SIZE = batch_size
        self.PREFETCH_SIZE = prefetch_size
        self.OVERLAP = overlap
        self.relative_joints = relative_joints
        self.variable_seqs = variable_seqs

        self.MAX_SEQ_LEN = seq_len ## 40 for example
        self.MIN_SEQ_LEN = seq_len // 2 # so like 20
        if self.relative_joints:
          self.OBS_DIM  = dataset[0]['obs'].shape[-1] + dataset[0]['joint_poses'].shape[-1] 
        else:
          self.OBS_DIM = dataset[0]['obs'].shape[-1]
        if self.relative_joints:
            self.ACT_DIM = dataset[0]['target_poses'].shape[-1] + 1 # +1 for the gripper
        else:
            self.ACT_DIM = dataset[0]['acts'].shape[-1]
            
        self.GOAL_DIM = dataset[0]['achieved_goals'].shape[-1] # 2 objects (xyz quat) + 4 uni dim = 18

        self.random_obj = random.Random(seed)

    def create_goal(self, trajectory, ti, tf):
        return np.tile(trajectory['achieved_goals'][tf, :], (tf-ti,1)) # Be wary of changing this, the planner relies on the fact that final goal is tiled out.

    def traj_to_subtrajs(self, trajectory, idx):
        """
        Converts a T-length trajectory into M subtrajectories of length SEQ_LEN, pads time dim to SEQ_LEN
        """
        T = len(trajectory['obs'])
        frame_skip = max(int(self.MAX_SEQ_LEN*self.OVERLAP),1)
        obs, goals, acts, masks, stored_seq_lens = [], [], [], [], [] # to save us compute time don't calc the seq len from the mask in training. 
        for ti in range(0,T-self.MAX_SEQ_LEN,frame_skip):
            if self.variable_seqs:
              seq_lens = list(self.MAX_SEQ_LEN - np.arange(0, self.MAX_SEQ_LEN-self.MIN_SEQ_LEN, 4)) #create an array of SEQ LENS with spacing 4
              
            else:
              seq_lens = [self.MAX_SEQ_LEN]
            
            for seq_len in seq_lens:
              tf = ti + seq_len
                  
              pad_len = self.MAX_SEQ_LEN-(tf-ti)
              time_padding = ((0,pad_len),(0,0))
              
              if self.relative_joints:
                  rel = trajectory['target_poses'][ti:tf] - trajectory['joint_poses'][ti:tf, :7]
                  gripper = np.expand_dims(trajectory['acts'][ti:tf, -1], -1)
                  action = np.pad(np.concatenate([rel, gripper], -1), time_padding)
                  o = np.concatenate([trajectory['obs'][ti:tf,:],trajectory['joint_poses'][ti:tf,:]],-1).astype('float32')
              else:
                  action = np.pad(trajectory['acts'][ti:tf], time_padding)
                  o = trajectory['obs'][ti:tf,:]
              
              obs.append(np.pad(o, time_padding))
              goals.append(np.pad(self.create_goal(trajectory, ti, tf), time_padding))
              acts.append(action)
              masks.append(np.pad(np.ones(tf-ti), time_padding[0]))
              stored_seq_lens.append([seq_len])
        
        return np.stack(obs), np.stack(goals), np.stack(acts), np.stack(masks), np.stack(stored_seq_lens)

    def create_tf_ds(self, is_training=True):
        """ Converts raw dataset to a shuffled subtraj dataset """
        dataset = self._train_data if is_training else self._valid_data
        obs, goals, acts, masks, seq_lens = [], [], [], [], []
        total_number_of_subtrajs = 0
        for idx, train_sample in enumerate(dataset):
            o,g,a,m,sl = self.traj_to_subtrajs(train_sample, idx)
            obs.append(o)
            goals.append(g)
            acts.append(a)
            masks.append(m)
            
            seq_lens.append(sl)
            total_number_of_subtrajs += len(o)

        obs = np.vstack(obs)
        goals = np.vstack(goals)
        acts = np.vstack(acts).astype('float32')
        masks = np.vstack(masks).astype('float32')
        seq_lens = np.vstack(seq_lens).astype('float32')
        print(f'Created {total_number_of_subtrajs} subtrajs')
        ds = tf.data.Dataset.from_tensor_slices((obs, goals, acts, masks, seq_lens)) 
        # Always shuffle, repeat then batch (in that order)
        ds = ds.shuffle(len(obs))
        ds = ds.repeat()
        ds = ds.batch(self.BATCH_SIZE, drop_remainder=True)
        ds = ds.prefetch(self.PREFETCH_SIZE)
        # ds = ds.cache()
        return ds
    
    
############## Everything from here for pybullet style #################################
def load_data(path, keys):
    cnt = Counter()
    dataset = []
    obs_act_path = os.path.join(path, 'obs_act_etc/')

    for demo in os.listdir(obs_act_path):
        
        traj = np.load(obs_act_path+demo+'/data.npz')
        traj = {key:traj[key] for key in keys}
        reset_states = []
        for i in range(0, len(traj[keys[0]])):
            # these are needed for deterministic resetting
            reset_states.append(path+'/states_and_ims/'+demo+'/env_states/'+str(i)+'.bullet')
        traj['reset_states'] = reset_states
        traj['reset_idx'] = int(demo)
        dataset.append(traj)
        
        cnt[len(traj[keys[0]])]+=1
        
    return dataset,cnt