#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/sholtodouglas/learning_from_play/blob/master/notebooks/languageXplay.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


# In[20]:
# import comet_ml at the top of your file
from comet_ml import Experiment
import wandb
wandb.login()

import argparse



parser = argparse.ArgumentParser(description='LFP training arguments')
parser.add_argument('run_name')
parser.add_argument('--train_datasets', nargs='+', help='Training dataset names')
parser.add_argument('--test_datasets', nargs='+', help='Testing dataset names')
parser.add_argument('-c', '--colab', default=False, action='store_true', help='Enable if using colab environment')
parser.add_argument('-s', '--data_source', default='DRIVE', help='Source of training data')
parser.add_argument('-tfr', '--from_tfrecords', default=False, action='store_true', help='Enable if using tfrecords format')
parser.add_argument('-d', '--device', default='TPU', help='Hardware device to train on')
parser.add_argument('-b', '--batch_size', default=512, type=int)
parser.add_argument('-wmax', '--window_size_max', default=40, type=int)
parser.add_argument('-wmin', '--window_size_min', default=20, type=int)
parser.add_argument('-la', '--actor_layer_size', default=2048, type=int, help='Layer size of actor, increases size of neural net')
parser.add_argument('-le', '--encoder_layer_size', default=512, type=int, help='Layer size of encoder, increases size of neural net')
parser.add_argument('-lp', '--planner_layer_size', default=2048, type=int, help='Layer size of planner, increases size of neural net')
parser.add_argument('-lg', '--goal_mapper_layer_size', default=512, type=int, help='Layer size of goal mapping networks from im and sent to goal space, increases size of neural net')
parser.add_argument('-embd', '--img_embedding_size', default=64, type=int, help='Embedding size of features,goal space')
parser.add_argument('-s_embd', '--sentence_embedding_size', default=512, type=int, help='Embedding size of MUSE sentence embeddings')
parser.add_argument('-g_embd', '--gripper_img_embedding_size', default=32, type=int, help='Embedding size of features,goal space')
parser.add_argument('-z', '--latent_dim', default=256, type=int, help='Size of the VAE latent space')
parser.add_argument('-zg', '--goal_space_dim', default=32, type=int, help='Size of the goal embedding space')
parser.add_argument('-g', '--gcbc', default=False, action='store_true', help='Enables GCBC, a simpler model with no encoder/planner')
parser.add_argument('-n', '--num_distribs', default=None, type=int, help='Number of distributions to use in logistic mixture model')
parser.add_argument('-q', '--qbits', default=None, type=int, help='Number of quantisation bits to discrete distributions into. Total quantisations = 2**qbits')
parser.add_argument('-lr', '--learning_rate', type=float, default=3e-4)
parser.add_argument('-t', '--train_steps', type=int, default=200000)
parser.add_argument('-r', '--resume', default=False, action='store_true')
parser.add_argument('-B', '--beta', type=float, default=0.00003)
parser.add_argument('-i', '--images', default=False, action='store_true')
parser.add_argument('-gi', '--gripper_images', default=False, action='store_true')
parser.add_argument('-sim', '--sim', default='Unity', help='Unity/Pybullet')
parser.add_argument('-vq', '--discrete', default=False, action='store_true')
parser.add_argument('-nm', '--normalize', default=False, action='store_true')
parser.add_argument('-lang', '--use_language', default=False, action='store_true')
parser.add_argument('-cont', '--use_contrastive', default=False, action='store_true')
parser.add_argument('-sub', '--sub_out_language_percent',  type=float, default=0.25)
parser.add_argument('--fp16', default=False, action='store_true')
parser.add_argument('--bucket_name', help='GCS bucket name to stream data from')
parser.add_argument('--tpu_name', help='GCP TPU name') # Only used in the script on GCP

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
# -lr 1e-3

print(args)

from pathlib import Path
from pathy import Pathy
import os
import requests
import json
import pprint
import logging
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import time
pp = pprint.PrettyPrinter(indent=4)

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
    if args.colab:
      auth.authenticate_user()
    print('Reading data from Google Cloud Storage')
    r = requests.get('https://ipinfo.io')
    region = r.json()['region']
    project_id = 'learning-from-play-303306'
    logging.warning(f'You are accessing GCS data from {region}, make sure this is the same as your bucket {args.bucket_name}')
    STORAGE_PATH = Pathy(f'gs://{args.bucket_name}')
else:
    print('Reading data from local filesystem')
    STORAGE_PATH = WORKING_PATH

print(f'Storage path: {STORAGE_PATH}')
TRAIN_DATA_PATHS = [STORAGE_PATH/'data'/x for x in args.train_datasets]
TEST_DATA_PATHS = [STORAGE_PATH/'data'/x for x in args.test_datasets]


# # Data Creation

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
    if args.fp16:
        tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
else:
    physical_devices = tf.config.list_physical_devices()
    if args.device == 'GPU':
        tf.config.experimental.set_memory_growth(physical_devices[3], enable=True)
        if args.fp16:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
    strategy = tf.distribute.get_strategy()
    NUM_DEVICES = 1
    print(physical_devices)

# # Dataset

GLOBAL_BATCH_SIZE = args.batch_size * NUM_DEVICES
dl = lfp.data.PlayDataloader(normalize=args.normalize, include_imgs = args.images, include_gripper_imgs = args.gripper_images, sim=args.sim, batch_size=GLOBAL_BATCH_SIZE,  window_size=args.window_size_max, min_window_size=args.window_size_min)

# Train data
train_data = dl.extract(TRAIN_DATA_PATHS, from_tfrecords=args.from_tfrecords)
train_dataset = dl.load(train_data)

# Validation data
valid_data = dl.extract(TEST_DATA_PATHS, from_tfrecords=args.from_tfrecords)
valid_dataset = dl.load(valid_data)


# # Model

# # Training Loop

from lfp.train import LFPTrainer


if args.device=='CPU' or args.device=='GPU':
     actor, encoder, planner, cnn, gripper_cnn,  img_embed_to_goal_space, lang_embed_to_goal_space, trainer =lfp.train.train_setup(args, dl, GLOBAL_BATCH_SIZE, strategy=None)
else:
    with strategy.scope():
         actor, encoder, planner, cnn, gripper_cnn,  img_embed_to_goal_space, lang_embed_to_goal_space, trainer = lfp.train.train_setup(args, dl, GLOBAL_BATCH_SIZE, strategy=None)
        
        
train_dist_dataset = iter(strategy.experimental_distribute_dataset(train_dataset))
valid_dist_dataset = iter(strategy.experimental_distribute_dataset(valid_dataset))

plotting_background_dataset = iter(valid_dataset) #for the cluster fig, easier with a non distributed dataset
# For use with lang and plotting the colored dots
labelled_dl = lfp.data.labelled_dl(batch_size=64)
labelled_test_ds = iter(labelled_dl.load(labelled_dl.extract(TEST_DATA_PATHS)))

if args.use_language:
    lang_dl = lfp.data.labelled_dl(batch_size=GLOBAL_BATCH_SIZE, shuffle_size = 64) # this is probably fine as it is preshuffled during creation
    train_dist_lang_dataset = iter(strategy.experimental_distribute_dataset(lang_dl.load(lang_dl.extract(TRAIN_DATA_PATHS))))
    valid_dist_lang_dataset = iter(strategy.experimental_distribute_dataset(lang_dl.load(lang_dl.extract(TEST_DATA_PATHS))))


from tensorflow.keras.utils import Progbar
progbar = Progbar(args.train_steps, verbose=1, interval=0.5)
valid_inc = 20
save_inc = 2000


run_name = args.run_name
model_path = str(STORAGE_PATH/'saved_models'/args.run_name)

if args.resume:
  # WandB reinit
  with open(f'{model_path}/config.json', 'r') as f:
      data = json.load(f)
  # Comet ML reinit
  exp = ExistingExperiment(api_key="C4vcCM57bnSYEsdncguxDW8pO",  previous_experiment=data['experiment_key'])

  wandb.init(project="learning-from-play_v2", id=data['run_id'],  resume="must")
  t = wandb.run.step + valid_inc # Todo get this from comet to complete the transition

  load_weights(model_path, actor, encoder, planner, with_optimizer=True)
  print('Loaded model weights and optimiser state')
  
  prognar.add(t, []) # update the progbar to the most recent point
else:
  #Comet
  experiment = Experiment(api_key="C4vcCM57bnSYEsdncguxDW8pO",project_name="learning-from-play",workspace="sholtodouglas")
  experiment.set_name(run_name)
  # WandB
  wandb.init(project="learning-from-play_v2")
  wandb.run.name = run_name
  t = 0

from lfp.metric import log # gets state and clears simultaneously

# Autograph just
# Creating these autograph wrappers so that tf.data operations are executed in graph mode
#@tf.function
def train(dataset, beta):
    train_batch = next(dataset)
    train_lang_batch = next(train_dist_lang_dataset) if args.use_language else None
    trainer.distributed_train_step(train_batch, train_lang_batch, beta)

#@tf.function
def test(dataset, beta):
    valid_batch = next(dataset)
    valid_lang_batch = next(valid_dist_lang_dataset) if args.use_language else None
    trainer.distributed_test_step(valid_batch, valid_lang_batch, beta)

while t < args.train_steps:
    start_time = time.time()
    train(train_dist_dataset, args.beta)

    if t % valid_inc == 0:
        test(valid_dist_dataset, args.beta)
        step_time = round(time.time() - start_time, 1)

        metrics = {metric_name: log(metric) for metric_name, metric in trainer.metrics.items()}
        metrics['step_time'] = step_time

        # validation plotting
        progbar.add(valid_inc, [('Train Loss', metrics['train_loss']),
                                ('Validation Loss', metrics['valid_loss']),
                                ('Time (s)', step_time)])
        #Plot on Comet
        experiment.log_metrics(metrics,step=t)
        # Plot on WandB
        wandb.log(metrics, step=t)

    if (t+0) % save_inc == 0: # zero while we test this
        trainer.save_weights(model_path, run_id=wandb.run.id, experiment_key=experiment.get_key())

        # How we plot the cluster figs
        batches = [trainer.make_sequences_variable_length(plotting_background_dataset.next()) for i in range(0,4)]
        super_batch = {}
        for k in batches[0].keys():
            super_batch[k] = np.concatenate([b[k] for b in batches])
        lang_batch = labelled_test_ds.next()
        fig_enc, fig_plan, z_enc, z_plan = lfp.plotting.produce_cluster_fig(super_batch, lang_batch, trainer, args=args)
        #if not args.gcbc and not args.images:
        #   z_enc, z_plan = produce_cluster_fig(next(plotting_dataset), encoder, planner, TEST_DATA_PATHS[0], num_take=dl.batch_size//4)

        #   #Comet
        #   experiment.log_figure('z_enc', z_enc, step=t)
        #   experiment.log_figure('z_plan', z_plan,step=t)

        # WandB
        wandb.log({'z_enc':lfp.plotting.plot_to_image(fig_enc), 'z_plan':lfp.plotting.plot_to_image(fig_plan)}, step=t)

          #latent_fig = project_enc_and_plan(ze, zp)
          #latent_img = plot_to_image(latent_fig)

    t += 1

