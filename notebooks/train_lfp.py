#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/sholtodouglas/learning_from_play/blob/master/notebooks/languageXplay.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


# In[20]:


import argparse

parser = argparse.ArgumentParser(description='LFP training arguments')
parser.add_argument('run_name')
parser.add_argument('--train_datasets', nargs='+', help='Training dataset names')
parser.add_argument('--test_datasets', nargs='+', help='Testing dataset names')
parser.add_argument('-c', '--colab', default=False, action='store_true', help='Enable if using colab environment')
parser.add_argument('-s', '--data_source', default='LOCAL', choices=['LOCAL', 'DRIVE', 'GCS'], help='Source of training data')
parser.add_argument('-tfr', '--from_tfrecords', default=False, action='store_true', help='Enable if using tfrecords format')
parser.add_argument('-d', '--device', default='CPU', choices=['CPU', 'GPU', 'TPU'],  help='Hardware device to train on')
parser.add_argument('-b', '--batch_size', default=32, type=int)
parser.add_argument('-la', '--actor_layer_size', default=256, type=int, help='Layer size of actor, increases size of neural net')
parser.add_argument('-le', '--encoder_layer_size', default=256, type=int, help='Layer size of encoder, increases size of neural net')
parser.add_argument('-lp', '--planner_layer_size', default=256, type=int, help='Layer size of planner, increases size of neural net')
parser.add_argument('-z', '--latent_dim', default=32, type=int, help='Size of the VAE latent space')
parser.add_argument('-g', '--gcbc', default=False, action='store_true', help='Enables GCBC, a simpler model with no encoder/planner')
parser.add_argument('-n', '--num_distribs', default=None, type=int, help='Number of distributions to use in logistic mixture model')
parser.add_argument('-q', '--qbits', default=None, type=int, help='Number of quantisation bits to discrete distributions into. Total quantisations = 2**qbits')
parser.add_argument('-lr', '--learning_rate', type=float, default=3e-4)
parser.add_argument('-t', '--train_steps', type=int, default=100000)
parser.add_argument('-r', '--resume', default=False, action='store_true')

parser.add_argument('--bucket_name', help='GCS bucket name to stream data from')
parser.add_argument('--tpu_name', help='GCP TPU name')

args = parser.parse_args()

# Argument validation
if args.device == 'TPU' and args.data_source == 'GCS':
    if args.bucket_name is None or args.tpu_name is None:
        parser.error('When using GCP TPUs you must specify the bucket and TPU names')

# python3 notebooks/train_lfp.py \
# tpuv3-test \
# --train_dataset UR5 UR5_slow_gripper UR5_high_transition \
# --test_dataset UR5_slow_gripper_test \
# -tfr \
# -s GCS \
# -d TPU \
# -b 512 \
# -la 2048 \
# -le 512 \
# -lp 2048 \
# -z 256 \
# -lr 3e-4

print(args)


# In[3]:


from pathlib import Path
from pathy import Pathy
import os
import requests
import json
import pprint
import logging
import numpy as np
import tensorflow as tf
import time
import wandb

pp = pprint.PrettyPrinter(indent=4)

wandb.login()


# In[4]:


#@title Workpace Setup (Local vs Colab)

# Set up working directory and libraries
if args.colab:
    from google.colab import drive, auth
    print('Using colab setup')
    WORKING_PATH = Path('/content/learning_from_play')
    # Clone repo
    try:
        get_ipython().system("git clone 'https://github.com/sholtodouglas/learning_from_play' {WORKING_PATH}")
    except: 
        pass
    # Mount drive
    drive.mount('/content/drive')
else:
    print('Using local setup')
    WORKING_PATH = Path.cwd()
    print(f'Working path: {WORKING_PATH}')

# Change working directory to learning_from_play
os.chdir(WORKING_PATH)
import lfp
        
# Set up storage directory and datasets
if args.data_source == 'DRIVE':
    assert args.colab, "Must be using Colab"
    print('Reading data from Google Drive')
    STORAGE_PATH = Path('/content/drive/My Drive/Robotic Learning')
elif args.data_source == 'GCS':
    print('Reading data from Google Cloud Storage')
    r = requests.get('https://ipinfo.io')
    region = r.json()['region']
    project_id = 'learning-from-play-303306'
    #     if region != 'Iowa':
    logging.warning(f'You are accessing GCS data from {region}, make sure this is the same as your bucket {args.bucket_name}')
    STORAGE_PATH = Pathy(f'gs://{args.bucket_name}')
else:
    print('Reading data from local filesystem')
    STORAGE_PATH = WORKING_PATH

print(f'Storage path: {STORAGE_PATH}')
TRAIN_DATA_PATHS = [STORAGE_PATH/'data'/x for x in args.train_datasets]
TEST_DATA_PATHS = [STORAGE_PATH/'data'/x for x in args.test_datasets]


# # Data Creation

# In[46]:


print("Tensorflow version " + tf.__version__)

if args.device == 'TPU':
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=args.tpu_name)  # TPU detection
        print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
    except ValueError:
        raise BaseException('ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')

    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)

    NUM_DEVICES = strategy.num_replicas_in_sync
    print("REPLICAS: ", NUM_DEVICES)
else:
    physical_devices = tf.config.list_physical_devices()
    if args.device == 'GPU':
        tf.config.experimental.set_memory_growth(physical_devices[3], enable=True)
    NUM_DEVICES = 1
    print(physical_devices)


# In[ ]:


# Use this to edit modules without needing to restart the kernel
# !git pull
# import importlib
# importlib.reload(lfp.data)
# importlib.reload(lfp.model)
# importlib.reload(lfp.plotting)
# importlib.reload(lfp.train)


# # Dataset

# ### Config Flags

# In[48]:


GLOBAL_BATCH_SIZE = args.batch_size * NUM_DEVICES

dl = lfp.data.PlayDataloader(batch_size=GLOBAL_BATCH_SIZE)
dl.shuffle_size = GLOBAL_BATCH_SIZE * 1 # May need to be smaller with images - lets test


# In[49]:


# Train data
train_data = dl.extract(TRAIN_DATA_PATHS, from_tfrecords=args.from_tfrecords)
train_dataset = dl.load(train_data)


# In[50]:


# Validation data
valid_data = dl.extract(TEST_DATA_PATHS, from_tfrecords=args.from_tfrecords)
valid_dataset = dl.load(valid_data)


# # Model

# # Training Loop

# In[51]:


def train_setup():
    model_params = {'obs_dim':dl.obs_dim,
                    'goal_dim':dl.goal_dim,
                    'act_dim':dl.act_dim,
                    'latent_dim':args.latent_dim}
    
    actor = lfp.model.create_actor(layer_size=args.actor_layer_size, gcbc=args.gcbc, num_distribs=args.num_distribs, **model_params)

    if args.gcbc:
        encoder = None
        planner = None
    else:
        encoder = lfp.model.create_encoder(layer_size=args.encoder_layer_size, **model_params)
        planner = lfp.model.create_planner(layer_size=args.planner_layer_size, **model_params)
        
    trainer = lfp.train.LFPTrainer(dl, 
                                 actor,
                                 encoder=encoder, 
                                 planner=planner, 
                                 probabilistic=args.num_distribs is not None,
                                 distribute_strategy=strategy,
                                 learning_rate=args.learning_rate,
                                 clipnorm=1.0,
                                 gcbc=args.gcbc)
    
    return trainer

if args.device=='GPU' or args.device == 'CPU':
    trainer = train_setup()
    train_dataset = iter(train_dataset)
    valid_dataset = iter(valid_dataset)
    plotting_dataset = valid_dataset # For consistnecy with the distributed form 
else:
    with strategy.scope():
        trainer = train_setup()   
    train_dataset = iter(strategy.experimental_distribute_dataset(train_dataset))
    plotting_dataset = iter(valid_dataset) # for the cluster fig, easier with a non distributed dataset
    valid_dataset = iter(strategy.experimental_distribute_dataset(valid_dataset))


# In[52]:


# 0.00001 so far works best for MAE - try lower - 0.00003 with a BETA accel of 10 so far works best, perfect encoder, nicely mapped planner
# recall 0.01 worked okay for probabilistic - proba still too unstable!
beta_sched = lfp.train.BetaScheduler('linear', 
                                   beta=0.00003, 
                                   beta_max=0.00003, 
                                   max_steps=args.train_steps, 
                                   cycles=90, 
                                   duty_cycle=0.5)


# In[53]:


progbar = tf.keras.utils.Progbar(args.train_steps, verbose=1, interval=0.5)
best_valid_loss = np.float('inf')

valid_inc = 20
save_inc = 1000

prev_grad_norm = np.float('inf')


# In[54]:


model_path = f'/content/drive/My Drive/Robotic Learning/saved_models/{args.run_name}/'

if args.resume:
    with open(model_path+'/config.json') as json_file:
        data = json.load(json_file)
        wandb.init(project="learning-from-play_v2", id=data['run_id'], resume="must")
        trainer.load_weights(model_path, with_optimizer=True)
        print('Loaded model weights and optimiser state')
        t = wandb.run.step + valid_inc
else:
    wandb.init(project="learning-from-play_v2", config=args)
    wandb.run.name = args.run_name
    t = 0


# In[ ]:


while t < args.train_steps:
    start_time = time.time()
    beta = beta_sched.scheduler(t)
    x = next(train_dataset)
    total_train_loss = trainer.distributed_train_step(x, beta)
    
    if t % valid_inc == 0:  
        valid_x = next(valid_dataset)
        if args.gcbc:
            total_val_loss, metrics = trainer.distributed_test_step(valid_x, beta)
        else:
            total_val_loss, metrics, ze, zp = trainer.distributed_test_step(valid_x, beta)

        # validation plotting
        progbar.add(valid_inc, [('Train Loss', metrics['train_loss']), 
                                ('Validation Loss', metrics['valid_loss']), 
                                ('Time (s)', round(time.time() - start_time, 1))])

        wandb.log(metrics, step=t)
          
    if t % save_inc == 0:
        trainer.save_weights(model_path, args, wandb.run.id)
        if not args.gcbc:
            z_enc, z_plan = lfp.plotting.produce_cluster_fig(next(plotting_dataset), trainer.encoder, trainer.planner, TEST_DATA_PATHS[0], num_take=dl.batch_size)
            convergence_plot = lfp.plotting.project_enc_and_plan(ze, zp)
            wandb.log({'z_enc':z_enc, 'z_plan':z_plan, 'convergence_plot':convergence_plot}, step=t)
    t += 1


# In[ ]:




