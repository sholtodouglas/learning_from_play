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
parser.add_argument('--bulk_datasets', nargs='+', help='data diversity dataset names')
parser.add_argument('--video_datasets', nargs='+', help='for contrastive learning')
parser.add_argument('-c', '--colab', default=False, action='store_true', help='Enable if using colab environment')
parser.add_argument('-s', '--data_source', default='GCS', help='Source of training data')
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
parser.add_argument('-enc_all', '--encode_all', default=False, action='store_true', help='encode_actions_and_proprio not just imgs')
parser.add_argument('-sub', '--sub_out_language_percent',  type=float, default=0.25)
parser.add_argument('--fp16', default=False, action='store_true')
parser.add_argument('--bucket_name', help='GCS bucket name to stream data from')
parser.add_argument('--tpu_name', help='GCP TPU name') # Only used in the script on GCP
# Set these to split the dataset up so we control the proportion of lang vs bulk vs video etc - make them batch numbers 
parser.add_argument('-ss', '--standard_split', type=int, default=0)
parser.add_argument('-bs', '--bulk_split', type=int, default=0)
parser.add_argument('-ls', '--lang_split', type=int, default=0)
parser.add_argument('-vs', '--video_split', type=int, default=0)
parser.add_argument('--init_from', type=str, default="")
args = parser.parse_args()

# Argument validation
if args.device == 'TPU' and args.data_source == 'GCS':
    if args.bucket_name is None or args.tpu_name is None:
        parser.error('When using GCP TPUs you must specify the bucket and TPU names')

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

print('Using local setup')
WORKING_PATH = Path.cwd()
print(f'Working path: {WORKING_PATH}')


print("Tensorflow version " + tf.__version__)

if args.device == 'TPU':
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=args.tpu_name)  # TPU detection
        print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
    except ValueError:
        raise BaseException('ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')

    print("Connect to cluster")
    # tf.config.experimental_connect_to_cluster(tpu)
    # print("Initialise")
    # tf.tpu.experimental.initialize_tpu_system(tpu)
    print("Strat")
    strategy = tf.distribute.TPUStrategy(tpu)
    print("NUMDEV")
    NUM_DEVICES = strategy.num_replicas_in_sync
    print("REPLICAS: ", NUM_DEVICES)
    if args.fp16:
        tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
