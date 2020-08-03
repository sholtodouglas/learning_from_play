import glob
import pickle
import random
from collections import Counter
import subprocess
import shlex
from multiprocessing.pool import ThreadPool
import tensorflow as tf

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