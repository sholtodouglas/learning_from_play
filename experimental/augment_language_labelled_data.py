
import argparse

parser = argparse.ArgumentParser(description='LFP training arguments')
parser.add_argument('--teleop_datasets', nargs='+', help='Training dataset names')
parser.add_argument('--video_datasets', nargs='+', help='for contrastive learning')
parser.add_argument('-s', '--data_source', default='GCS', help='Source of training data')
parser.add_argument('--bucket_name', help='GCS bucket name to stream data from')
parser.add_argument('-t', '--steps', type=int, default=3000)
args = parser.parse_args()

print(args)
import sys
sys.path.append("..") # Adds higher directory to python modules path.

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
import random 
import lfp

pp = pprint.PrettyPrinter(indent=4)



WORKING_PATH = Path().absolute().parent
os.chdir(WORKING_PATH)
print(os.getcwd())


print(f'Working path: {WORKING_PATH}')

if args.data_source == 'GCS':
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
TRAIN_DATA_PATHS = [STORAGE_PATH/'data'/x for x in args.teleop_datasets]
VIDEO_DATA_PATHS = [STORAGE_PATH/'data'/x for x in args.video_datasets] if args.video_datasets != None else []

import tensorflow_hub as hub
if args.data_source == 'GCS':
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    model = hub.load(module_url)
    print ("module %s loaded" % module_url)
    def embed(input):
      return model(input)
else:
    embed = hub.KerasLayer(str(STORAGE_PATH)+'/saved_models/universal_sentence_encoder')
    
    
synonyms = [
     ['push', 'depress', 'hit', 'press', 'prod', 'tap'],
    ['switch', 'button'],
    ['prism', 'rectangle', 'block'],
    ['grab', 'grasp', 'seize'],
    ['pick the block'],
    ['cupboard', 'cabinet'],
    ['place', 'put'],
    ['pull', 'drag']
]

def subout(label):
    string = label.numpy().decode()
    for replacements in synonyms:
        for word in replacements:
            if word in string:
                replacement = random.choice(replacements)
                string = string.replace(word, replacement)
    return string
    
def augment(t, serialise):
    t['tag'] = t['tags']
    t['label'] = subout(t['labels'])
    print(t['label'])
    t['label_embedding'] =  np.squeeze(embed([t['label']]))
    return serialise(t)


from lfp.data import read_traj_tfrecord, serialise_traj
from tqdm import tqdm      
from lfp.data import read_vid, serialise_vid

def create_ds(PATHS, read, write):
    labelled_dl = lfp.data.labelled_dl(batch_size=1, read_func=read)
    for path in PATHS:
        label_it = iter(labelled_dl.extract([path]).repeat())
        buff = { i:label_it.next() for i in range(0,100)} # implement this as a dict, so we can randomly choose an element, then replace that - i.e like a tf shuffle buffer
        save_path = str(path/'tf_records')+f"/labelled_augmented.tfrecords"
        with tf.io.TFRecordWriter(save_path) as file_writer:
            print(save_path)
            for i in tqdm(range(0,args.steps)):
                
                choice = random.randint(0, len(buff)-1)
                
                byte_stream = augment(buff[choice], write)
                file_writer.write(byte_stream)
                buff[choice] = label_it.next() # replace the one we used with a new random one, use dict not list as o(1) insertion /deletion


create_ds(TRAIN_DATA_PATHS, read_traj_tfrecord, serialise_traj)

create_ds(VIDEO_DATA_PATHS, read_vid, serialise_vid)
