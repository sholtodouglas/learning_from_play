import glob
import tensorflow as tf
import os
import numpy as np
from natsort import natsorted
import sklearn
import pprint
from tqdm import tqdm

pp = pprint.PrettyPrinter(indent=4)


def get_tf_records(PATHS, bucket_name):
    record_paths = []
    folder_names = ["data/"+str(p) for p in PATHS]
    for p in folder_names: # pass a list of folders and it will find the tf records in them
        record_paths += tf.io.gfile.glob(f"gs://{bucket_name}/{p}/tf_records/*")
    return record_paths

# TF record specific @ Tristan maybe we can clean this by having the one dict and a function which handles which parse to use?
def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.reshape(image, [200,200, 3]) # explicit size needed for TPU
    return image

def read_tfrecord(example):
    LABELED_TFREC_FORMAT = {
            'obs': tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring,
            'acts': tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring,
            'achieved_goals': tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring,
            'joint_poses': tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring,
            'target_poses': tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring,
            'acts_quat': tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring,
            'acts_rpy_rel': tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring,
            'velocities': tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring,
            'obs_quat': tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring,
            'proprioception': tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring,
            'sequence_index': tf.io.FixedLenFeature([], tf.int64),
            'sequence_id': tf.io.FixedLenFeature([], tf.int64),
            'img': tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring,
    }
    data = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    
    obs = tf.io.parse_tensor(data['obs'], tf.float32) 
    acts = tf.io.parse_tensor(data['acts'], tf.float32) 
    achieved_goals = tf.io.parse_tensor(data['achieved_goals'], tf.float32)  
    joint_poses = tf.io.parse_tensor(data['joint_poses'], tf.float32)  
    target_poses = tf.io.parse_tensor(data['target_poses'], tf.float32) 
    acts_quat =tf.io.parse_tensor( data['acts_quat'], tf.float32) 
    acts_rpy_rel = tf.io.parse_tensor(data['acts_rpy_rel'], tf.float32)  
    velocities = tf.io.parse_tensor(data['velocities'], tf.float32) 
    obs_quat = tf.io.parse_tensor(data['obs_quat'], tf.float32) 
    proprioception = tf.io.parse_tensor(data['proprioception'], tf.float32)  
    sequence_index = tf.cast(data['sequence_index'], tf.int32) 
    sequence_id = tf.cast(data['sequence_id'], tf.int32) # this is meant to be 32 even though you serialize as 64
    
    img = decode_image(data['img'])

    return {'obs' : obs, 
            'acts' : acts, 
            'achieved_goals' : achieved_goals, 
            'joint_poses' : joint_poses, 
            'target_poses' : target_poses, 
            'acts_quat' : acts_quat, 
            'acts_rpy_rel' : acts_rpy_rel, 
            'velocities' : velocities, 
            'obs_quat' : obs_quat, 
            'proprioception' : proprioception, 
            'sequence_index' : sequence_index, 
            'sequence_id' : sequence_id,
            'img' : img}

def extract_npz(paths):
    keys = ['obs', 'acts', 'achieved_goals', 'joint_poses', 'target_poses', 'acts_quat', 'acts_rpy_rel',
            'velocities', 'obs_quat', 'proprioception']
    dataset = {k: [] for k in keys + ['sequence_index', 'sequence_id']}

    for path in paths:
        obs_act_path = os.path.join(path, 'obs_act_etc/')
        for demo in tqdm(natsorted(os.listdir(obs_act_path)), desc=path.name):
            traj = np.load(obs_act_path + demo + '/data.npz')
            for k in keys:
                d = traj[k]
                if len(d.shape) < 2:
                    d = np.expand_dims(d, axis=1)  # was N, should be N,1
                dataset[k].append(d.astype(np.float32))
            timesteps = len(traj['obs'])
            dataset['sequence_index'].append(np.arange(timesteps, dtype=np.int32).reshape(-1, 1))
            dataset['sequence_id'].append(np.full(timesteps, fill_value=int(demo), dtype=np.int32).reshape(-1, 1))

    # convert to numpy
    for k in keys + ['sequence_index', 'sequence_id']:
        dataset[k] = np.vstack(dataset[k])
    # convert to tf dataset
    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    return dataset

def extract_tfrecords(paths, ordered=True, num_workers=tf.data.experimental.AUTOTUNE):
    # In our case, order does matter
    tf_options = tf.data.Options()
    tf_options.experimental_deterministic = ordered  # disable order, increase speed

    dataset = tf.data.TFRecordDataset(paths, num_parallel_reads=num_workers)
    dataset = dataset.with_options(tf_options)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=num_workers)
    return dataset


class PlayDataloader():
    """
    Usage:
    1. From npz (i.e. without images)
        dataset = dataloader.extract(paths)
        dataset = dataloader.load(dataset)
    2. From tf_records (i.e. with images)
        dataset = dataloader.extract(paths, from_tfrecords=True)
        dataset = dataloader.load(dataset)
    """
    def __init__(self, 
                relative_obs=False,
                relative_act=False,
                quaternion_act=False,
                joints=False,
                velocity=False,
                normalize=False,
                proprioception=False,
                batch_size=32, # 512*8 with the TPU
                window_size=50,
                min_window_size=20,
                window_shift=1,
                variable_seqs=True, 
                num_workers=tf.data.experimental.AUTOTUNE,
                seed=42):
        
        self.relative_obs = relative_obs
        self.relative_act = relative_act
        self.quaternion_act = quaternion_act
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
        self.mean_obs, self.std_obs, self.mean_acts, self.standard_acts = None, None, None,None
        # Todo redo the standardisation find original version here https://github.com/sholtodouglas/learning_from_play/blob/9f385c0c80f905da63b9954e192dac696559e163/languageXplay.ipynb

    @staticmethod
    def print_minutes(dataset):
        dataset_size = sum(1 for _ in dataset)
        secs = dataset_size / 25
        hours = secs // 3600
        minutes = secs // 60 - hours * 60
        print(f"{dataset_size} frames, which is {hours:.0f}hrs {minutes:.0f}m.")

    def create_goal_tensor(self, dataset, seq_len=-1):
        ''' Tile final achieved_goal across time dimension '''
        tile_dims = tf.constant([self.window_size, 1], tf.int32)
        goal = tf.tile(dataset['achieved_goals'][seq_len-1,tf.newaxis], tile_dims) # as goal is at an index take seq_len -1
        return goal

    def validate_action_label(self, acts):
        if self.quaternion_act or self.joints:
            raise tf.errors.NotImplementedError
        else:
            action_limits = tf.constant([1.5, 1.5, 2.2, 3.2, 3.2, 3.2, 1.1])
            tf.debugging.Assert(tf.logical_and(tf.reduce_all(-action_limits < acts),
                                               tf.reduce_all(action_limits > acts)),
                                data=[acts],
                                name="act_limit_validation")
        
    def extract(self, paths, from_tfrecords=False):
        """

        :param paths:
        :param from_tfrecords:
        :return:
        """
        if from_tfrecords:
            if '.tfrecords' in paths[0]: 
                record_paths = paths # we've already got the paths because we are direct GCS streaming
            else:
                record_paths = []
                for p in paths:
                    record_paths += glob.glob(str(p/'tf_records/*.tfrecords'))
            dataset = extract_tfrecords(record_paths, ordered=True, num_workers=self.num_workers)
        else:
            dataset = extract_npz(paths)
        # self.print_minutes(dataset)
        # self.validate_labels(dataset)
        return dataset
    
    def transform(self, dataset):
        """

        :param dataset:
        :return:
        """
        # Action limit validation
        self.validate_action_label(dataset['acts'])

        # State representations
        obs = dataset['obs']
        
        # act in joint space
        if self.joints:
            obs = tf.concat([obs,dataset['joint_poses'][:,:7]], axis=-1)
            acts = dataset['target_poses']
            if self.relative_act:
                acts = acts - dataset['joint_poses'][:,:6]

        # act in position space with quaternions
        elif self.quaternion_act:
                acts = dataset['acts_quat'][:,:7]
                if self.relative_act:
                    acts = acts  - dataset['obs_quat'][:,:7]
        # act in rpy position space            
        else:
            acts = dataset['acts'][:,:6]
            if self.relative_act:
                acts = acts - dataset['obs'][:,:6]
                
        # add the gripper on the end
        gripper = dataset['acts'][:,-1,tf.newaxis]
        acts = tf.concat([acts, gripper], axis=-1)

        if self.relative_obs:
            obs = np.diff(obs, axis=0)
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
            # Record the mean like we used to @Tristan
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
        """

        :param dataset: a tf Dataset
        :return:
        """
            
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
        pp.pprint(dataset.element_spec)
        return dataset