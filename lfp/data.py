import glob
import tensorflow as tf
import os
import numpy as np
from natsort import natsorted
import sklearn
import pprint
from tqdm import tqdm

pp = pprint.PrettyPrinter(indent=4)

import glob
import tensorflow as tf
import os
import numpy as np
from natsort import natsorted
import sklearn
import pprint
from tqdm import tqdm

pp = pprint.PrettyPrinter(indent=4)

dimensions = {'Unity': {'obs': 19,
                        'acts': 7,
                        'achieved_goals': 12,
                        'shoulder_img_hw':256,
                        'hz': 15},
              'Pybullet': {'obs': 18,
                        'acts': 7,
                        'achieved_goals': 11,
                        'shoulder_img_hw':200,
                        'hz': 25}}


# TF record specific @ Tristan maybe we can clean this by having the one dict and a function which handles which parse to use?
def decode_shoulder_img(image_data, image_hw=256):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.reshape(image, [image_hw,image_hw, 3]) # explicit size needed for TPU
    return image

def decode_gripper_img(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.reshape(image, [64,64, 3]) # explicit size needed for TPU
    return image

def read_tfrecord(include_imgs=False, include_gripper_imgs=False, sim='Unity'):
    def read_tfrecord_helper(example):
        LABELED_TFREC_FORMAT = {
            'obs': tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring,
            'acts': tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring,
            'achieved_goals': tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring,
            'sequence_index': tf.io.FixedLenFeature([], tf.int64),
            'sequence_id': tf.io.FixedLenFeature([], tf.int64),
        }
        if include_imgs:
            LABELED_TFREC_FORMAT['img'] = tf.io.FixedLenFeature([], tf.string) # tf.string means bytestring
        if include_gripper_imgs:
            LABELED_TFREC_FORMAT['gripper_img'] = tf.io.FixedLenFeature([], tf.string) # tf.string means bytestring

        data = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

        output = {}
        output['obs'] = tf.ensure_shape(tf.io.parse_tensor(data['obs'], tf.float32), (dimensions[sim]['obs'],))
        output['acts'] = tf.ensure_shape(tf.io.parse_tensor(data['acts'], tf.float32), (dimensions[sim]['acts'],))
        output['achieved_goals'] = tf.ensure_shape(tf.io.parse_tensor(data['achieved_goals'], tf.float32), (dimensions[sim]['achieved_goals'],))
        output['sequence_index'] = tf.cast(data['sequence_index'], tf.int32)
        output['sequence_id'] = tf.cast(data['sequence_id'], tf.int32) # this is meant to be 32 even though you serialize as 64
        if include_imgs:
            output['img'] = decode_shoulder_img(data['img'], dimensions[sim]['shoulder_img_hw'])
        if include_gripper_imgs:
            output['gripper_img'] = decode_gripper_img(data['gripper_img'])

        return output
    return read_tfrecord_helper

def extract_npz(paths):
    keys = ['obs', 'acts', 'achieved_goals']
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

def extract_tfrecords(paths, include_imgs=False, include_gripper_imgs=False, sim='Unity', ordered=True, num_workers=1):
    # In our case, order does matter
    tf_options = tf.data.Options()
    tf_options.experimental_deterministic = ordered  # must be 1 to maintain order while streaming from GCS

    dataset = tf.data.TFRecordDataset(paths, num_parallel_reads=1)
    dataset = dataset.with_options(tf_options)
    dataset = dataset.map(read_tfrecord(include_imgs, include_gripper_imgs, sim), num_parallel_calls=num_workers)
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
                gripper_proprioception=False,
                batch_size=32,
                window_size=40,
                min_window_size=20,
                window_shift=1,
                include_imgs=False,
                include_gripper_imgs=False,
                shuffle_size=None, 
                num_workers=tf.data.experimental.AUTOTUNE,
                seed=42,
                sim='Unity'):
        
        self.relative_obs = relative_obs
        self.relative_act = relative_act
        self.quaternion_act = quaternion_act
        self.joints = joints
        self.velocity = velocity
        self.normalize = normalize
        self.gripper_proprioception = gripper_proprioception
        
        self.batch_size = batch_size
        self.window_size = window_size
        self.window_shift = window_shift
        self.include_imgs = include_imgs
        self.include_gripper_imgs = include_gripper_imgs
        self.shuffle_size = int(batch_size * (window_size / window_shift)) if shuffle_size is None else shuffle_size
        self.prefetch_size = tf.data.experimental.AUTOTUNE
        self.num_workers = num_workers
        self.seed = seed
        self.mean_obs, self.std_obs, self.mean_acts, self.standard_acts = None, None, None,None
        self.sim=sim
        # Todo redo the standardisation find original version here https://github.com/sholtodouglas/learning_from_play/blob/9f385c0c80f905da63b9954e192dac696559e163/languageXplay.ipynb

    @staticmethod
    def print_minutes(dataset, sim):
        dataset_size = sum(1 for _ in dataset)
        secs = dataset_size / dimensions[sim]['hz']
        hours = secs // 3600
        minutes = secs // 60 - hours * 60
        print(f"{dataset_size} frames, which is {hours:.0f}hrs {minutes:.0f}m.")


    def extract(self, paths, from_tfrecords=False):
        """

        :param paths:
        :param include_imgs:
        :param from_tfrecords:
        :return:
        """
        if from_tfrecords:
            record_paths = []
            for p in paths:
                record_paths += tf.io.gfile.glob(str(p/'tf_records/*.tfrecords'))
            dataset = extract_tfrecords(record_paths, self.include_imgs, self.include_gripper_imgs, self.sim, ordered=True, num_workers=self.num_workers)
        else:
            dataset = extract_npz(paths)
        # self.print_minutes(dataset, self.sim)
        return dataset
    
    def transform(self, dataset):
        """

        :param dataset:
        :return:
        """
        # State representations
        obs = dataset['obs']
        
        # act in joint space
        acts = dataset['acts']

        return_dict = {'obs': obs,
                'acts': acts,
                'goals': dataset['achieved_goals'], # use this for the goal tiling on device
                'dataset_path': dataset['sequence_id'],
                'tstep_idxs': dataset['sequence_index']}

        if self.include_imgs:
            return_dict['imgs'] = dataset['img'] # use this for the goal tiling on device
            # Proprioceptive features are xyz, rpy, gripper angle
            return_dict['proprioceptive_features'] = obs[:7]
        if self.include_gripper_imgs:
            return_dict['gripper_imgs'] = dataset['gripper_img']

        # TODO: Tristan with images we may not want to return the normal goals/states at all  just straight sub out
        return return_dict

    # Note: main reasons the dataloader was slow before are:
    # - mapping transform after windowing (we want it ideally before)
    # - var seq lens and goal tiling are computationally expensive (we ought to do on device)
    # - single-threaded tfrecords mapper (NOT the IO)
    # - action validation also taxing, it could be risky removing but it's also too slow. We should validate our
    #   data once off beforehand. Todo: write a data validator
    def load(self, dataset):
        """

        :param dataset: a tf Dataset
        :return:
        """
        window_lambda = lambda x: tf.data.Dataset.zip(x).batch(self.window_size)
        seq_overlap_filter = lambda x: tf.equal(tf.size(tf.unique(tf.squeeze(x['dataset_path'])).y), 1)
        dataset = (dataset
                    .map(self.transform, num_parallel_calls=self.num_workers)
                    .window(size=self.window_size, shift=self.window_shift, stride=1, drop_remainder=True)
                    .flat_map(window_lambda)
                    .filter(seq_overlap_filter) # Todo: optimise this/remove if possible
                    .repeat()
                    .shuffle(self.shuffle_size)
                    .batch(self.batch_size, drop_remainder=True)
                    .prefetch(self.prefetch_size))

        self.obs_dim = dataset.element_spec['obs'].shape[-1]
        self.goal_dim = dataset.element_spec['goals'].shape[-1]
        self.act_dim = dataset.element_spec['acts'].shape[-1]
        if self.include_imgs:
            self.img_size = dataset.element_spec['imgs'].shape[-2]
            self.proprioceptive_features_dim = dataset.element_spec['proprioceptive_features'].shape[-1]
        if self.include_gripper_imgs:
            self.gripper_img_size = dataset.element_spec['gripper_imgs'].shape[-2]

        pp.pprint(dataset.element_spec)
        return dataset