import glob
import tensorflow as tf
import os
import numpy as np
from natsort import natsorted
import sklearn
import pprint
from tqdm import tqdm

pp = pprint.PrettyPrinter(indent=4)

from io import BytesIO
from tensorflow.python.lib.io import file_io
import lfp.unity_utils as uu
import random

dimensions = {'Unity': {'obs': 19,
                        'obs_extra_info': uu.messaging.UNITY_MAX_OBS_SIZE,
                        'acts': 7,
                        'achieved_goals': 12, 
                        'achieved_goals_extra_info':uu.messaging.UNITY_MAX_AG_SIZE,
                        'shoulder_img_hw':200,
                        'hz': 25},
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

def read_tfrecord(include_imgs=False,  include_imgs2 = False, include_gripper_imgs=False, sim='Unity'):
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
        if include_imgs2:
            LABELED_TFREC_FORMAT['img2'] = tf.io.FixedLenFeature([], tf.string) # tf.string means bytestring
        if include_gripper_imgs:
            LABELED_TFREC_FORMAT['gripper_img'] = tf.io.FixedLenFeature([], tf.string) # tf.string means bytestring

        data = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

        output = {}
        if include_imgs:
            output['obs'] = tf.ensure_shape(tf.io.parse_tensor(data['obs'], tf.float32), (dimensions[sim]['obs_extra_info'],))
            output['achieved_goals'] = tf.ensure_shape(tf.io.parse_tensor(data['achieved_goals'], tf.float32), (dimensions[sim]['achieved_goals_extra_info'],))
        else:
            output['obs'] = tf.ensure_shape(tf.io.parse_tensor(data['obs'], tf.float32), (dimensions[sim]['obs'],))
            output['achieved_goals'] = tf.ensure_shape(tf.io.parse_tensor(data['achieved_goals'], tf.float32), (dimensions[sim]['achieved_goals'],))

        output['acts'] = tf.ensure_shape(tf.io.parse_tensor(data['acts'], tf.float32), (dimensions[sim]['acts'],))
        output['sequence_index'] = tf.cast(data['sequence_index'], tf.int32)
        output['sequence_id'] = tf.cast(data['sequence_id'], tf.int32) # this is meant to be 32 even though you serialize as 64
        if include_imgs:
            output['img'] = decode_shoulder_img(data['img'], dimensions[sim]['shoulder_img_hw'])
        if include_imgs2:
            output['img2'] = decode_shoulder_img(data['img2'], dimensions[sim]['shoulder_img_hw'])
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

def extract_tfrecords(paths, include_imgs=False, include_imgs2 = False, include_gripper_imgs=False, sim='Unity', ordered=True, num_workers=1):
    # In our case, order does matter
    tf_options = tf.data.Options()
    tf_options.experimental_deterministic = ordered  # must be 1 to maintain order while streaming from GCS

    dataset = tf.data.TFRecordDataset(paths, num_parallel_reads=1)
    dataset = dataset.with_options(tf_options)
    dataset = dataset.map(read_tfrecord(include_imgs, include_imgs2, include_gripper_imgs, sim), num_parallel_calls=num_workers)
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
                include_imgs2=False,
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
        self.include_imgs2 = include_imgs2
        self.include_gripper_imgs = include_gripper_imgs
        self.shuffle_size = int(batch_size * (window_size / window_shift))*10 if shuffle_size is None else shuffle_size
        self.prefetch_size = tf.data.experimental.AUTOTUNE
        self.num_workers = num_workers
        self.seed = seed
        self.mean_obs, self.std_obs, self.mean_acts, self.standard_acts = None, None, None,None
        self.sim=sim
        if self.sim == 'Unity' and not self.include_imgs:
            print('Confirm that data dimensions are correct - states will be cut down to exclude additional objects introduced for the image dataset - states has only been test on blue block.')

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
                records = tf.io.gfile.glob(str(p/'tf_records/*.tfrecords'))
                record_paths += [pth for pth in records if 'label' not in pth]
            random.shuffle(record_paths) # to ensure good mixing of different time periods of data (important in the bulk dataset)
            dataset = extract_tfrecords(record_paths, self.include_imgs, self.include_imgs2, self.include_gripper_imgs, self.sim, ordered=True, num_workers=self.num_workers)
        else:
            dataset = extract_npz(paths)
        # self.print_minutes(dataset, self.sim)
        if self.normalize:
            src = str(paths[0])+'/normalisation.npz'
            f = BytesIO(file_io.read_file_to_string(src, binary_mode=True))
            self.normalising_constants = np.load(f)

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

        ags = dataset['achieved_goals']

        if not self.include_imgs: # If its from states, apply the max state dim to cut out excess info we might have room for many objects from states
            obs = obs[:dimensions[self.sim]['obs']]
            ags = ags[:dimensions[self.sim]['achieved_goals']]

        if self.normalize:
            obs = obs - self.normalising_constants['obs_mean']
            acts = acts - self.normalising_constants['acts_mean']
            ags = ags - self.normalising_constants['ag_mean']


        return_dict = {'obs': obs,
                'acts': acts,
                'goals': ags, # use this for the goal tiling on device
                'dataset_path': dataset['sequence_id'],
                'tstep_idxs': dataset['sequence_index']}

        if self.include_imgs:
            return_dict['imgs'] = dataset['img'] # use this for the goal tiling on device
            return_dict['proprioceptive_features'] = obs[:7]
        if self.include_imgs2:
            return_dict['imgs2'] = dataset['img2'] # use this for the goal tiling on device
            # Proprioceptive features are xyz, rpy, gripper angle
            
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
    def load(self, dataset, batch_size=None):
        """

        :param dataset: a tf Dataset
        :return:
        """
        if batch_size == None:
            batch_size = self.batch_size

        window_lambda = lambda x: tf.data.Dataset.zip(x).batch(self.window_size)
        seq_overlap_filter = lambda x: tf.equal(tf.size(tf.unique(tf.squeeze(x['dataset_path'])).y), 1)
        dataset = (dataset
                    .map(self.transform, num_parallel_calls=self.num_workers)
                    .window(size=self.window_size, shift=self.window_shift, stride=1, drop_remainder=True)
                    .flat_map(window_lambda)
                    .filter(seq_overlap_filter) # Todo: optimise this/remove if possible
                    .repeat()
                    .shuffle(self.shuffle_size)
                    .batch(batch_size, drop_remainder=True)
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





#########################################################################################################################################################################################################
################################################################### For reading datasets which are made of full trajectories (e.g the sentence labelled trajectories) ###################################
#########################################################################################################################################################################################################
from tensorflow.train import BytesList, FloatList, Int64List
from tensorflow.train import Example, Features, Feature


def serialise_traj(data):
    
    # obs, acts, goals, seq_lens, masks, imgs ,imgs2, gripper_imgs, goal_imgs, proprioceptive_features, label, label_embedding, tag  = data['obs'], \
    # data['acts'], data['goals'], data['seq_lens'], data['masks'], data['imgs'], data['imgs2'], data['gripper_imgs'], data['goal_imgs'], data['goal_imgs2'], data['proprioceptive_features'], data['label'], data['label_embedding'], data['tag']
    
    # obs (1, 40, 19)
    # acts (1, 40, 7)
    # goals (1, 40, 11)
    # seq_lens (1,)
    # masks (1, 40)
    # imgs (1, 1, 200, 200, 3)
    # imgs (1, 1, 200, 200, 3)
    # gripperimgs (1, 40, 64, 64, 3)
    # goal_imgs (1, 1, 128, 128, 3)
    # proprioceptive_features (1, 40, 7)
    # label string
    # label embedding [1,512]
    # tag string

    # obs = Feature(bytes_list=BytesList(value=[tf.io.serialize_tensor(obs).numpy(),]))
    # acts = Feature(bytes_list=BytesList(value=[tf.io.serialize_tensor(acts).numpy(),]))
    # goals = Feature(bytes_list=BytesList(value=[tf.io.serialize_tensor(goals).numpy(),]))
    # seq_lens = Feature(int64_list=Int64List(value=[seq_lens,]))
    # masks = Feature(bytes_list=BytesList(value=[tf.io.serialize_tensor(masks).numpy(),])) 

    # imgs = Feature(bytes_list=BytesList(value=[tf.io.serialize_tensor(imgs).numpy(),]))
    # imgs2 = Feature(bytes_list=BytesList(value=[tf.io.serialize_tensor(imgs2).numpy(),]))
    # gripper_imgs = Feature(bytes_list=BytesList(value=[tf.io.serialize_tensor(gripper_imgs).numpy(),]))
    # goal_imgs = Feature(bytes_list=BytesList(value=[tf.io.serialize_tensor(goal_imgs).numpy(),]))
    # goal_imgs2 = Feature(bytes_list=BytesList(value=[tf.io.serialize_tensor(goal_imgs2).numpy(),]))
    # proprioceptive_features = Feature(bytes_list=BytesList(value=[tf.io.serialize_tensor(proprioceptive_features).numpy(),]))
    # label =  Feature(bytes_list=BytesList(value=[tf.io.serialize_tensor(label).numpy(),]))
    # label_embedding = Feature(bytes_list=BytesList(value=[tf.io.serialize_tensor(label_embedding).numpy(),]))
    # tag =  Feature(bytes_list=BytesList(value=[tf.io.serialize_tensor(tag).numpy(),]))
    
    # features = Features(feature={
    #           'obs':obs,
    #           'acts':acts,
    #           'goals':goals,
    #           'seq_lens':seq_lens,
    #           'masks':masks,
    #           'imgs':imgs,
    #           'imgs2':imgs2,
    #           'gripper_imgs':gripper_imgs,
    #           'goal_imgs':goal_imgs,
    #           'goal_imgs2':goal_imgs2,
    #           'proprioceptive_features':proprioceptive_features,
    #           'label': label,
    #           'label_embedding': label_embedding,
    #           'tag': tag})
    
    # example = Example(features=features)

    features = {k: Feature(bytes_list=BytesList(value=[tf.io.serialize_tensor(v).numpy(),])) for k,v in data.items() if k not in ['seq_lens']}
    features['seq_lens'] =  Feature(int64_list=Int64List(value=[data['seq_lens'],]))

    example = Example(features=Features(feature=features))
    
    
    return example.SerializeToString()

def read_traj_tfrecord(example):
    LABELED_TFREC_FORMAT = {
            'obs':tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring,
            'acts':tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring,
            'goals':tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring,
            'seq_lens':tf.io.FixedLenFeature([], tf.int64),
            'masks':tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring,
            'imgs':tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring,
            'imgs2':tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring,
            'gripper_imgs':tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring,
            'goal_imgs':tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring,
            'proprioceptive_features':tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring,
            'label':tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring,
            'label_embedding':tf.io.FixedLenFeature([], tf.string),
            'tag': tf.io.FixedLenFeature([], tf.string),
    }
    data = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    
    obs = tf.io.parse_tensor(data['obs'], tf.float32) 
    acts = tf.io.parse_tensor(data['acts'], tf.float32) 
    goals = tf.io.parse_tensor(data['goals'], tf.float32)  
    seq_lens = tf.cast(data['seq_lens'], tf.int32) # this is meant to be 32 even though you serialize as 64
    masks = tf.io.parse_tensor(data['masks'], tf.float32) 
    imgs = tf.io.parse_tensor(data['imgs'], tf.uint8)
    imgs2 = tf.io.parse_tensor(data['imgs2'], tf.uint8)
    gripper_imgs = tf.io.parse_tensor(data['gripper_imgs'], tf.uint8)
    goal_imgs = tf.io.parse_tensor(data['goal_imgs'], tf.uint8)     
    proprioceptive_features =tf.io.parse_tensor( data['proprioceptive_features'], tf.float32) 
    label = tf.io.parse_tensor(data['label'], tf.string)
    label_embedding = tf.io.parse_tensor(data['label_embedding'], tf.float32)
    tag = tf.io.parse_tensor(data['tag'], tf.string)
    


    
    # img = decode_image(data['img'])

    return {  'obs':obs,
              'acts':acts,
              'goals':goals,
              'seq_lens':seq_lens,
              'masks':masks,
              'imgs':imgs,
              'imgs2':imgs2,
              'gripper_imgs':gripper_imgs,
              'goal_imgs':goal_imgs,
              'proprioceptive_features':proprioceptive_features,
              'labels': label,
              'label_embeddings':label_embedding,
              'tags': tag}


def load_traj_tf_records(filenames, read_func, ordered=False):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # disregarding data order. Order does not matter since we will be shuffling the data anyway.

    # check, does this ignore intra order or just inter order? Both are an issue!
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=tf.data.experimental.AUTOTUNE) # 
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(read_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset


class labelled_dl():
        def __init__(self,
            label_type='label',  # 'tag' for numbered types or 'labels'
            include_images=True,
            sim = 'Unity',
            num_workers=4,
            batch_size=64,
            shuffle_size=512,
            normalize=False,
            read_func=read_traj_tfrecord):

            self.num_workers = num_workers
            self.include_images = include_images
            self.sim = 'Unity'
            self.batch_size = batch_size
            self.shuffle_size = shuffle_size
            self.normalize = normalize
            self.label_type = label_type
            self.read_func = read_func

            if self.normalize:
                raise NotImplementedError
                # src = str(filenames[0])+'/normalisation.npz'
                # f = BytesIO(file_io.read_file_to_string(src, binary_mode=True))
                # self.normalising_constants = np.load(f)

            self.prefetch_size = tf.data.experimental.AUTOTUNE

            
        def extract(self, filenames):
            labelled_paths = []
            for p in filenames:
                records = tf.io.gfile.glob(str(p/'tf_records/*.tfrecords'))
                labelled_paths += [pth for pth in records if self.label_type  in pth]

            return load_traj_tf_records(labelled_paths, self.read_func)

        def load(self, dataset, batch_size=None):
            if batch_size == None:
                batch_size = self.batch_size

            dataset = (dataset
                        .map(self.transform, num_parallel_calls=self.num_workers)
                        .repeat()
                        .shuffle(self.shuffle_size)
                        .batch(batch_size, drop_remainder=True)
                        .prefetch(self.prefetch_size))
            return dataset


        
        def transform(self, dataset):
                # tile out the goal img

            if not self.include_images: # If its from states, apply the max state dim to cut out excess info we might have room for many objects from states
                obs = dataset['obs'][:dimensions[self.sim]['obs']]
                ags = dataset['goals'][:dimensions[self.sim]['achieved_goals']]

            return dataset

################################################################### Serialise just videos
###################################################################
def serialise_vid(data):
    
    # seq_lens, masks, imgs, goal_imgs,label, label_embedding, tag = data['seq_lens'], data['masks'], data['imgs'], data['goal_imgs'], data['label'], data['label_embedding'], data['tag']
    
    features = {k: Feature(bytes_list=BytesList(value=[tf.io.serialize_tensor(v).numpy(),])) for k,v in data.items() if k not in ['seq_lens']}
    features['seq_lens'] =  Feature(int64_list=Int64List(value=[data['seq_lens'],]))

    example = Example(features=Features(features))
    
    
    return example.SerializeToString()
    # # seq_lens (1,)
    # # masks (1, 40)
    # # imgs (1, 1, 128, 128, 3)
    # # goal_imgs (1, 1, 128, 128, 3)
    # # label string
    # # label embedding [1,512]
    # # tag string

    # seq_lens = Feature(int64_list=Int64List(value=[seq_lens,]))
    # masks = Feature(bytes_list=BytesList(value=[tf.io.serialize_tensor(masks).numpy(),])) 
    # imgs = Feature(bytes_list=BytesList(value=[tf.io.serialize_tensor(imgs).numpy(),]))
    # goal_imgs = Feature(bytes_list=BytesList(value=[tf.io.serialize_tensor(goal_imgs).numpy(),]))
    # label =  Feature(bytes_list=BytesList(value=[tf.io.serialize_tensor(label).numpy(),]))
    # label_embedding = Feature(bytes_list=BytesList(value=[tf.io.serialize_tensor(label_embedding).numpy(),]))
    # tag =  Feature(bytes_list=BytesList(value=[tf.io.serialize_tensor(tag).numpy(),]))
    
    # features = Features(feature={
    #           'seq_lens':seq_lens,
    #           'masks':masks,
    #           'imgs':imgs,
    #           'goal_imgs':goal_imgs,
    #           'label': label,
    #           'label_embedding': label_embedding,
    #           'tag': tag})
    
    # example = Example(features=features)
    
    # return example.SerializeToString()

def read_vid(example):
    LABELED_TFREC_FORMAT = {
            'seq_lens':tf.io.FixedLenFeature([], tf.int64),
            'masks':tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring,
            'imgs':tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring,
            'goal_imgs':tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring,
            'label':tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring,
            'label_embedding':tf.io.FixedLenFeature([], tf.string),
            'tag': tf.io.FixedLenFeature([], tf.string),
    }
    data = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    
    seq_lens = tf.cast(data['seq_lens'], tf.int32) # this is meant to be 32 even though you serialize as 64
    masks = tf.io.parse_tensor(data['masks'], tf.float32) 
    imgs = tf.io.parse_tensor(data['imgs'], tf.uint8)
    imgs2 = tf.io.parse_tensor(data['imgs2'], tf.uint8)
    goal_imgs = tf.io.parse_tensor(data['goal_imgs'], tf.uint8)
    label = tf.io.parse_tensor(data['label'], tf.string)
    label_embedding = tf.io.parse_tensor(data['label_embedding'], tf.float32)
    tag = tf.io.parse_tensor(data['tag'], tf.string)
    

    return {  'seq_lens':seq_lens,
              'masks':masks,
              'imgs':imgs,
              'imgs2':imgs2,
              'goal_imgs':goal_imgs,
              'labels': label,
              'label_embeddings':label_embedding,
              'tags': tag}

### Can use the labelled_dl with read_func = read_vid to read

# create dataloader which combines the  datasets, and outputs next - makes them distributed if necessary. 
# # This is so we can use TFrecord speed, but get the right proportions of various datasets (e.g a bulk pretraining one for extra data diversity)
class distributed_data_coordinator:
    # load

    def __init__(self,
        args,
        TRAIN_DATA_PATHS, # all data we have - these are lists of folder paths,
        TEST_DATA_PATHS,
        strategy,
        BULK_DATA_PATHS=[], # data we might want to emphaise in higher proportion - e.g if training for a standard viewpoint or environment
        VIDEO_DATA_PATHS=[],
        standard_split = 64,
        bulk_split = 0, # lets make these the actual number cause it needs to be 8 divisible - 
        lang_split = 0,
        video_split = 0,
        NUM_DEVICES = 8,
        ): # non-teleop, video only data

        self.args = args

        # bulk is like backup data that we won't have much of but enough for the diversity
        self.GLOBAL_BATCH_SIZE = args.batch_size * NUM_DEVICES
        # If we didn't set the split, assume everything in our main train/test DS
        if standard_split == 0: 
            standard_split = args.batch_size
        self.bulk_split, self.standard_split, self.lang_split, self.video_split = int(bulk_split* NUM_DEVICES), int(standard_split* NUM_DEVICES), int(lang_split* NUM_DEVICES), int(video_split* NUM_DEVICES)
        print(f"Our dataset split is {self.standard_split} specific, {self.lang_split} lang, {self.video_split} video, {self.bulk_split} bulk")
        assert (self.bulk_split+self.standard_split+self.lang_split+self.video_split) == self.GLOBAL_BATCH_SIZE
        if args.use_language: assert self.lang_split > 0
        
        ######################################### Train
        self.dl = PlayDataloader(normalize=args.normalize, include_imgs = args.images, include_imgs2 = args.images2, include_gripper_imgs = args.gripper_images, sim=args.sim,  window_size=args.window_size_max, min_window_size=args.window_size_min)
        self.dl_lang =  labelled_dl(sim = args.sim) # this is probably fine as it is preshuffled during creation
        self.standard_dataset =  iter(strategy.experimental_distribute_dataset(self.dl.load(self.dl.extract(TRAIN_DATA_PATHS, from_tfrecords=args.from_tfrecords),  batch_size=self.standard_split)))
        self.bulk_dataset =  iter(strategy.experimental_distribute_dataset(self.dl.load(self.dl.extract(BULK_DATA_PATHS, from_tfrecords=args.from_tfrecords), batch_size=self.bulk_split))) if self.bulk_split > 0 else None
        
        ######################################### Test
        valid_dataset = self.dl.load(self.dl.extract(TEST_DATA_PATHS, from_tfrecords=args.from_tfrecords), batch_size=self.bulk_split+self.standard_split)
        self.valid_dataset = iter(strategy.experimental_distribute_dataset(valid_dataset))
        
        ######################################### Plotting
        self.plotting_background_dataset = iter(self.dl.load(self.dl.extract(TEST_DATA_PATHS, from_tfrecords=args.from_tfrecords), batch_size=32)) #for the background in the cluster fig
        # For use with lang and plotting the colored dots
        tagged_dl = labelled_dl(label_type='tag', sim = args.sim)
        self.labelled_test_ds = iter(tagged_dl.load(tagged_dl.extract(TEST_DATA_PATHS)))

        ######################################### Language
        if args.use_language:
            self.lang_dataset =  iter(strategy.experimental_distribute_dataset(self.dl_lang.load(self.dl_lang.extract(TRAIN_DATA_PATHS+BULK_DATA_PATHS),  batch_size=self.lang_split)))
            self.lang_valid_dataset =  iter(strategy.experimental_distribute_dataset(self.dl_lang.load(self.dl_lang.extract(TEST_DATA_PATHS),  batch_size=self.lang_split)))
        else:
            train_dist_lang_dataset, valid_dist_lang_dataset = None, None
        
        ######################################### Contrastive
        if args.use_contrastive:
            raise NotImplementedError
        
    def next(self):
        batch = next(self.standard_dataset) 
        lang = next(self.lang_dataset) if self.args.use_language else tf.constant(0.0)  # uing 0 constants as distribute strat hates none
        video = next(self.video_dataset) if self.args.use_contrastive else tf.constant(0.0)  # uing 0 constants as distribute strat hates none
        bulk = next(self.bulk_dataset)  if self.bulk_split > 0 else tf.constant(0.0) # combine batch and standard on device
        return {'batch':batch, 'lang':lang, 'video':video, 'bulk':bulk}

    def next_valid(self):
        batch = next(self.valid_dataset)
        # no standard - just test whatever we are validiating against
        lang = next(self.lang_valid_dataset) if self.args.use_language else tf.constant(0.0)  # uing 0 constants as distribute strat hates none
        video = next(self.video_valid_dataset) if self.args.use_contrastive else tf.constant(0.0) # uing 0 constants as distribute strat hates none
        return {'batch': batch, 'lang': lang, 'video': video}



