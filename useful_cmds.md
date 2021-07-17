
# Available resources for TFRC

* 5 on-demand Cloud TPU v2-8 device(s) in zone us-central1-f
* 100 preemptible Cloud TPU v2-8 device(s) in zone us-central1-f
* 5 on-demand Cloud TPU v3-8 device(s) in zone europe-west4-a

- n2 (0.1) are default, e2-medium (0.033) work, g1-small do not work (0.02)
europe-west4-a, lfp_europe_west4_a



# Iowa (v2-8)
export TPU_ZONE=us-central1-f
export TPU_NAME=lfp-us1
export BUCKET_NAME=iowa_bucket_lfp
export TPU_SIZE=v2-8

# Europe (v3-8)
export TPU_ZONE=europe-west4-a
export TPU_SIZE=v3-8
export TPU_NAME=lfp1
export BUCKET_NAME=lfp_europe_west4_a
export PROJECT_ID=learning-from-play-303306



# Creating just a TPU-VM (new method)



gcloud alpha compute tpus tpu-vm create lfp8 --zone=europe-west4-a --accelerator-type=v3-8 --version=v2-alpha

gcloud alpha compute tpus tpu-vm ssh lfp7 --zone europe-west4-a --project learning-from-play-303306

gcloud alpha compute tpus tpu-vm delete lfp2 --zone=europe-west4-a



# Use tmux so that the process keeps running after ssh disconnect

# optionally clone the repo if not already there
# libTPU breaks with a normal TF installation
```
tmux
export BUCKET_NAME=lfp_europe_west4_a
git clone https://github.com/tensorflow/models.git
pip3 install -r models/official/requirements.txt

git clone https://github.com/sholtodouglas/learning_from_play

cd learning_from_play
./setup.sh
mkdir data
gsutil -m cp -r dir gs://$BUCKET_NAME/data/unity data
```
# Run the sample training script for GCS setup

```
python3 train_lfp.py \
tpuv3-test \
--train_dataset UR5 UR5_slow_gripper UR5_high_transition \
--test_dataset UR5_slow_gripper_test \
-tfr \
-s GCS \
-d TPU \
-b 512 \
-la 2048 \
-le 512 \
-lp 512 \
-z 256 \
-lr 3e-4 \
-B 0.00003 \
--bucket_name=$BUCKET_NAME \
--tpu_name=$TPU_NAME
```

**Note** if the ssh session disconnects it will kill the currently running python process.

To see which python processes are currently running use:

```ps -ef | grep python```


This the language model
"https://tfhub.dev/google/universal-sentence-encoder/4"

https://stackoverflow.com/questions/60578801/how-to-load-tf-hub-model-from-local-system



python3 train_lfp.py 2048_b00_lim_intensities_spatial_softmax --train_datasets unity/top_down_diverse_new --test_datasets unity/top_down_diverse_new_test -tfr -s LOCAL -d TPU -b 32 -la 2048 -le 2048 -lp 2048 -z 256 -lr 3e-4 -B 0.0 -n 5 -t 1000000 -wmin 20 -wmax 50 -i -gi -lang --bucket_name=$BUCKET_NAME --tpu_name=$TPU_NAME --standard_split 28 --lang_split 4 --bulk_split 0 --cnn intensities_spatial_softmax


python3 train_lfp.py \
debug \
--bulk_datasets unity/envHz25 \
--train_datasets unity/diverse \
--test_datasets unity/diverse_test \
-tfr \
-s GCS \
-d TPU \
-b 8 \
-la 32 \
-le 32 \
-lp 32 \
-z 8 \
-lr 3e-4 \
-B 0.02 \
-n 5 \
-t 1000000 \
-wmin 25 \
-wmax 50 \
-i \
-gi \
-lang \
--bucket_name=$BUCKET_NAME \
--tpu_name=$TPU_NAME \
--standard_split 4 \
--lang_split 2 \
--bulk_split 2
--cnn intensities_spatial_softmax


python3 augment_language_labelled_data.py --teleop_datasets unity/diverse --bucket_name lfp_europe_west4_a --video_datasets Unity/contrastive_vids



python3 train_lfp.py \
disc_1024_B1 \
--train_datasets unity/top_down_diverse_new  \
--test_datasets unity/top_down_diverse_new_test \
-tfr \
-s GCS \
-d TPU \
-b 32 \
-la 2048 \
-le 2048 \
-lp 2048 \
-z 1024 \
-lr 3e-4 \
-B 0.0 \
-n 5 \
-t 1000000 \
-wmin 25 \
-wmax 50 \
-i \
-gi \
-lang \
-vq \
-B 1 \
-tmp 1 \
--vq_tiles 5 \
--bucket_name=$BUCKET_NAME \
--tpu_name=$TPU_NAME \
--standard_split 28 \
--lang_split 4 \
--bulk_split 0 \
--cnn intensities_spatial_softmax


mkdir data
gsutil -m cp -r gs://$BUCKET_NAME/data/unity data
python3 train_lfp.py \
disc_4096_0.1 \
--train_datasets unity/top_down_diverse_new  \
--test_datasets unity/top_down_diverse_new_test \
-tfr \
-s GCS \
-d TPU \
-b 32 \
-la 2048 \
-le 2048 \
-lp 2048 \
-z 1024 \
-lr 3e-4 \
-B 0.0 \
-n 5 \
-t 1000000 \
-wmin 25 \
-wmax 50 \
-i \
-gi \
-lang \
-vq \
-tmp 0.1 \
--vq_tiles 5 \
--bucket_name=$BUCKET_NAME \
--tpu_name=$TPU_NAME \
--standard_split 28 \
--lang_split 4 \
--bulk_split 0 \
--cnn intensities_spatial_softmax

python3 train_lfp.py \
disc_1024_1 \
--train_datasets unity/top_down_diverse_new  \
--test_datasets unity/top_down_diverse_new_test \
-tfr \
-s GCS \
-d TPU \
-b 32 \
-la 2048 \
-le 2048 \
-lp 2048 \
-z 1024 \
-lr 3e-4 \
-B 0.0 \
-n 5 \
-t 1000000 \
-wmin 25 \
-wmax 50 \
-i \
-gi \
-lang \
-vq \
--temperature 1 \
--vq_tiles 5 \
--bucket_name=$BUCKET_NAME \
--tpu_name=$TPU_NAME \
--standard_split 28 \
--lang_split 4 \
--bulk_split 0 \
--cnn intensities_spatial_softmax



-enc_all \
--init_from sft_IMB001_lang_full_enc_v2


python3 train_lfp.py IMB002_lang_full_enc_bigCNNv2 --bulk_datasets unity/envHz25 --train_datasets unity/diverse --test_datasets unity/diverse_test -tfr -s GCS -d TPU -b 32 -la 2048 -le 512 -lp 2048 -z 256 -lr 3e-4 -B 0.02 -n 5 -t 1000000 -wmin 25 -wmax 50 -i -gi -lang --bucket_name=$BUCKET_NAME --tpu_name=$TPU_NAME --standard_split 16 --lang_split 8 --bulk_split 8 -enc_all --init_from IMB002_lang_full_enc_bigCNN


python3 train_lfp.py \
2048_b001_lim_impala \
--bulk_datasets unity/envHz25 unity/augmented_diverse_new \
--train_datasets unity/diverse unity/diverse_new \
--test_datasets unity/diverse_test \
-tfr \
-s LOCAL \
-d TPU \
-b 32 \
-la 2048 \
-le 2048 \
-lp 2048 \
-z 256 \
-lr 3e-4 \
-B 0.01 \
-n 5 \
-t 1000000 \
-wmin 25 \
-wmax 50 \
-i \
-gi \
-lang \
--bucket_name=$BUCKET_NAME \
--tpu_name=$TPU_NAME \
--standard_split 16 \
--lang_split 8 \
--bulk_split 8 \
--cnn impala

python3 train_lfp.py \
2048_b001_lim_deep_impala \
--bulk_datasets unity/envHz25 unity/augmented_diverse_new \
--train_datasets unity/diverse unity/diverse_new \
--test_datasets unity/diverse_test \
-tfr \
-s LOCAL \
-d TPU \
-b 32 \
-la 2048 \
-le 2048 \
-lp 2048 \
-z 256 \
-lr 3e-4 \
-B 0.01 \
-n 5 \
-t 1000000 \
-wmin 25 \
-wmax 50 \
-i \
-gi \
-lang \
--bucket_name=$BUCKET_NAME \
--tpu_name=$TPU_NAME \
--standard_split 16 \
--lang_split 8 \
--bulk_split 8 \
--cnn deep_impala

python3 train_lfp.py \
2048_b001_lim_intensities_spatial_softmax \
--train_datasets unity/augmented_diverse_new \
--test_datasets unity/augmented_diverse_new_test \
-tfr \
-s LOCAL \
-d TPU \
-b 32 \
-la 2048 \
-le 2048 \
-lp 2048 \
-z 256 \
-lr 3e-4 \
-B 0.01 \
-n 5 \
-t 1000000 \
-wmin 25 \
-wmax 50 \
-i \
-gi \
-lang \
--bucket_name=$BUCKET_NAME \
--tpu_name=$TPU_NAME \
--standard_split 28 \
--lang_split 4 \
--bulk_split 0 \
--cnn intensities_spatial_softmax


python3 train_lfp.py \
2048_b002_lim_intensities_spatial_softmax \
--bulk_datasets unity/envHz25 unity/augmented_diverse_new \
--train_datasets unity/diverse unity/diverse_new \
--test_datasets unity/diverse_test \
-tfr \
-s LOCAL \
-d TPU \
-b 32 \
-la 2048 \
-le 2048 \
-lp 2048 \
-z 256 \
-lr 3e-4 \
-B 0.02 \
-n 5 \
-t 1000000 \
-wmin 25 \
-wmax 50 \
-i \
-gi \
-lang \
--bucket_name=$BUCKET_NAME \
--tpu_name=$TPU_NAME \
--standard_split 16 \
--lang_split 8 \
--bulk_split 8 \
--cnn intensities_spatial_softmax

python3 train_lfp.py \
2048_b002_full_intensities_spatial_softmax \
--bulk_datasets unity/envHz25 unity/augmented_diverse_new \
--train_datasets unity/diverse unity/diverse_new \
--test_datasets unity/diverse_test \
-tfr \
-s LOCAL \
-d TPU \
-b 32 \
-la 2048 \
-le 2048 \
-lp 2048 \
-z 256 \
-lr 3e-4 \
-B 0.02 \
-n 5 \
-t 1000000 \
-wmin 25 \
-wmax 50 \
-i \
-gi \
-lang \
--bucket_name=$BUCKET_NAME \
--tpu_name=$TPU_NAME \
--standard_split 16 \
--lang_split 8 \
--bulk_split 8 \
--cnn intensities_spatial_softmax
-enc_all




bigenc_IMB002_lang_full_enc_v2, 1024, 0.02, 
super_enc, lim, 2048, 




# OLD
```
ctpu up \
--project=learning-from-play-303306 \
--zone=$TPU_ZONE \
--tf-version=2.4.1 \
--name=$TPU_NAME \
--tpu-size=$TPU_SIZE \
--machine-type=e2-medium \
--disk-size-gb=50

'''


mkdir data
gsutil -m cp -r gs://$BUCKET_NAME/data/unity data
python3 train_lfp.py \
2048_b001_lim_intensities_spatial_softmax \
--train_datasets unity/augmented_diverse_new \
--test_datasets unity/augmented_diverse_new_test \
-tfr \
-s LOCAL \
-d TPU \
-b 32 \
-la 2048 \
-le 2048 \
-lp 2048 \
-z 256 \
-lr 3e-4 \
-B 0.01 \
-n 5 \
-t 1000000 \
-wmin 25 \
-wmax 50 \
-i \
-gi \
-lang \
--bucket_name=$BUCKET_NAME \
--tpu_name=$TPU_NAME \
--standard_split 28 \
--lang_split 4 \
--bulk_split 0 \
--cnn intensities_spatial_softmax \
-i2


python3 train_lfp.py \
2048_b001_lim_intensities_spatial_softmax \
--train_datasets unity/augmented_diverse_new \
--test_datasets unity/augmented_diverse_new_test \
-tfr \
-s LOCAL \
-d TPU \
-b 32 \
-la 2048 \
-le 2048 \
-lp 2048 \
-z 256 \
-lr 3e-4 \
-B 0.02 \
-n 5 \
-t 1000000 \
-wmin 25 \
-wmax 50 \
-i \
-gi \
-lang \
--bucket_name=$BUCKET_NAME \
--tpu_name=$TPU_NAME \
--standard_split 28 \
--lang_split 4 \
--bulk_split 0 \
--cnn intensities_spatial_softmax \
-i2 \
-enc_all


python3 train_lfp.py \
2048_b001_lim_intensities_spatial_softmax \
--train_datasets pybullet/UR5 pybullet/UR5_high_transition pybullet/UR5_slow_gripper \
--test_datasets pybullet/UR5_slow_gripper_test \
-tfr \
-s LOCAL \
-d TPU \
-b 32 \
-la 2048 \
-le 2048 \
-lp 2048 \
-z 256 \
-lr 3e-4 \
-B 0.01 \
-n 5 \
-t 1000000 \
-wmin 20 \
-wmax 40 \
-i \
--bucket_name=$BUCKET_NAME \
--tpu_name=$TPU_NAME \
--standard_split 32 \
--lang_split 0 \
--bulk_split 0 \
--cnn intensities_spatial_softmax \
-enc_all

python3 train_lfp.py \
2048_b001_states \
--train_datasets pybullet/UR5 pybullet/UR5_high_transition pybullet/UR5_slow_gripper \
--test_datasets pybullet/UR5_slow_gripper_test \
-tfr \
-s LOCAL \
-d TPU \
-b 32 \
-la 2048 \
-le 2048 \
-lp 2048 \
-z 256 \
-lr 3e-4 \
-B 0.01 \
-n 5 \
-t 1000000 \
-wmin 20 \
-wmax 40 \
--bucket_name=$BUCKET_NAME \
--tpu_name=$TPU_NAME \
--standard_split 32 \
--lang_split 0 \
--bulk_split 0 \
-enc_all


mkdir data
gsutil -m cp -r gs://$BUCKET_NAME/data/unity data
python3 train_lfp.py \
disc_4096_0.1 \
--train_datasets unity/augmented_diverse_new \
--test_datasets unity/augmented_diverse_new_test \
-tfr \
-s GCS \
-d TPU \
-b 32 \
-la 2048 \
-le 2048 \
-lp 2048 \
-z 1024 \
-lr 3e-4 \
-B 0.0 \
-n 5 \
-t 1000000 \
-wmin 25 \
-wmax 50 \
-i \
-gi \
-lang \
-vq \
-tmp 0.1 \
--vq_tiles 5 \
--bucket_name=$BUCKET_NAME \
--tpu_name=$TPU_NAME \
--standard_split 28 \
--lang_split 4 \
--bulk_split 0 \
--cnn intensities_spatial_softmax