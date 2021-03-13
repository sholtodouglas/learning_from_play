

from tensorflow.train import BytesList, FloatList, Int64List
from tensorflow.train import Example, Features, Feature

def serialise(data):
    
    obs, acts, goals, seq_lens, masks, dataset_path, tstep_idxs , imgs , goal_imgs, proprioceptive_features = data['obs'], \
    data['acts'], data['goals'], data['seq_lens'], data['masks'], data['dataset_path'], data['tstep_idxs'], data['imgs'], data['goal_imgs'], data['proprioceptive_features']
    
    # obs (1, 40, 18)
    # acts (1, 40, 7)
    # goals (1, 40, 11)
    # seq_lens (1,)
    # masks (1, 40)
    # dataset_path (1, 40)
    # tstep_idxs (1, 40)
    # imgs (1, 40, 200, 200, 3)
    # goal_imgs (1, 40, 200, 200, 3)
    # proprioceptive_features (1, 40, 7)

    goal_imgs = tf.expand_dims(goal_imgs[:,0,:,:,:],1) # crete a :, 1, :,:,: shaped goal images for less file IO

    
    obs = Feature(bytes_list=BytesList(value=[tf.io.serialize_tensor(tf.squeeze(obs)).numpy(),]))
    acts = Feature(bytes_list=BytesList(value=[tf.io.serialize_tensor(tf.squeeze(acts)).numpy(),]))
    goals = Feature(bytes_list=BytesList(value=[tf.io.serialize_tensor(tf.squeeze(goals)).numpy(),])) 
    seq_lens = Feature(int64_list=Int64List(value=[seq_lens,]))
    masks = Feature(bytes_list=BytesList(value=[tf.io.serialize_tensor(tf.squeeze(masks)).numpy(),])) 

    imgs = Feature(bytes_list=BytesList(value=[tf.io.serialize_tensor(tf.squeeze(imgs)).numpy(),]))
    goal_imgs = Feature(bytes_list=BytesList(value=[tf.io.serialize_tensor(tf.squeeze(goal_imgs)).numpy(),]))
    proprioceptive_features = Feature(bytes_list=BytesList(value=[tf.io.serialize_tensor(tf.squeeze(proprioceptive_features)).numpy(),]))

    
    features = Features(feature={
              'obs':obs,
              'acts':acts,
              'goals':goals,
              'seq_lens':seq_lens,
              'masks':masks,
              'imgs':imgs,
              'goal_imgs':goal_imgs,
              'proprioceptive_features':proprioceptive_features})
    
    example = Example(features=features)
    
    return example.SerializeToString()




# Sample Usage
# r = lfp.data.PlayDataloader(include_imgs = args.images, batch_size=1,  window_size=args.window_size_max, min_window_size=args.window_size_min)
# rd = r.extract(TRAIN_DATA_PATHS, from_tfrecords=args.from_tfrecords)
# rd = r.load(rd)
# r_it = iter(rd)


# @tf.function
# def sample():
#   return r_it.next()

# data_paths = [str(STORAGE_PATH/'precompute')+f"/{x}.tfrecords" for x in range(0,8)]
# #@title write to gcs
# from tqdm import tqdm
# for path in data_paths:
#   with tf.io.TFRecordWriter(path) as file_writer:
#     print(path)
#     for i in tqdm(range(0,200)):
#         byte_stream = serialise(sample())
#         file_writer.write(byte_stream)