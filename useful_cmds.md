
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

gcloud alpha compute tpus tpu-vm create $TPU_NAME \
  --zone=$TPU_ZONE\
  --accelerator-type=$TPU_SIZE \
  --version=v2-alpha

gcloud alpha compute tpus create $TPU_NAME \
  --zone=$TPU_ZONE\
  --accelerator-type=$TPU_SIZE \
  --version=v2-alpha

# Creating TPU + VM

```
ctpu up \
--project=learning-from-play-303306 \
--zone=$TPU_ZONE \
--tf-version=2.4.1 \
--name=$TPU_NAME \
--tpu-size=$TPU_SIZE \
--machine-type=e2-medium \
--disk-size-gb=50


ctpu up \
--project=learning-from-play-303306 \
--zone=$TPU_ZONE \
--tf-version=2.4.1 \
--name=$TPU_NAME \
--tpu-size=$TPU_SIZE \
--disk-size-gb=50

[--preemptible]
```
Can also potentially opt for an `f1-micro` if we're still under-utilising the VM

# ssh into vm instance

See more info here: https://cloud.google.com/sdk/docs/quickstart

```gcloud compute ssh root@$TPU_NAME --zone=$TPU_ZONE```


# Use tmux so that the process keeps running after ssh disconnect

# optionally clone the repo if not already there
```
git clone https://github.com/sholtodouglas/learning_from_play

cd learning_from_play
./setup.sh
```


export BUCKET_NAME=lfp_europe_west4_a
export TPU_NAME=lfp1
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



python3 train_lfp.py \
PROB_IM_BIGPLAN_B0_01 \
--train_dataset UR5 UR5_slow_gripper UR5_high_transition \
--test_dataset UR5_slow_gripper_test \
-tfr \
-s GCS \
-d TPU \
-b 32 \
-la 2048 \
-le 512 \
-lp 2048 \
-z 256 \
-lr 3e-4 \
-B 0.01 \
-t 500000 \
-wmin 20 \
-wmax 40 \
-i \
-tfr \
-n 5 \
--bucket_name=$BUCKET_NAME \
--tpu_name=$TPU_NAME


python3 train_lfp.py \
IM_BIGPLAN_B0_00003 \
--train_dataset UR5 UR5_slow_gripper UR5_high_transition \
--test_dataset UR5_slow_gripper_test \
-tfr \
-s GCS \
-d TPU \
-b 16 \
-la 2048 \
-le 512 \
-lp 2048 \
-z 256 \
-lr 3e-4 \
-B 0.00003 \
-t 500000 \
-wmin 20 \
-wmax 40 \
-i \
-tfr \
--bucket_name=$BUCKET_NAME \
--tpu_name=$TPU_NAME


--train_dataset Unity/envHz25 \
--test_dataset Unity/envHz25_test \
--train_dataset Unity/serv12Hz \
--test_dataset Unity/serv12Hz_test \

python3 train_lfp.py \
300kstatesB0_04 \
--train_dataset Unity/envHz25 \
--test_dataset Unity/envHz25_test \
-tfr \
-s GCS \
-d TPU \
-b 512 \
-la 2048 \
-le 512 \
-lp 2048 \
-z 256 \
-lr 3e-4 \
-B 0.04 \
-n 5 \
-t 1000000 \
-wmin 20 \
-wmax 40 \
--bucket_name=$BUCKET_NAME \
--tpu_name=$TPU_NAME

python3 train_lfp.py \
300kImB0_02 \
--train_dataset Unity/envHz25 \
--test_dataset Unity/envHz25_test \
-tfr \
-s GCS \
-d TPU \
-b 32 \
-la 2048 \
-le 512 \
-lp 2048 \
-z 256 \
-lr 3e-4 \
-B 0.02 \
-n 5 \
-t 1000000 \
-wmin 20 \
-wmax 40 \
-i \
-gi \
--bucket_name=$BUCKET_NAME \
--tpu_name=$TPU_NAME



python3 train_lfp.py \
PybulletB0_02_2040_lim \
--train_dataset UR5 \
--test_dataset UR5_slow_gripper_test \
-tfr \
-s GCS \
-d TPU \
-b 512 \
-la 2048 \
-le 512 \
-lp 2048 \
-z 256 \
-lr 3e-4 \
-B 0.02 \
-n 5 \
-wmin 20 \
-wmax 40 \
-sim Pybullet \
--bucket_name=$BUCKET_NAME \
--tpu_name=$TPU_NAME


python3 train_lfp.py \
UNITY_IM_B0_01 \
--train_dataset Unity/UR5_train \
--test_dataset Unity/UR5_test \
-tfr \
-s GCS \
-d TPU \
-b 16 \
-la 2048 \
-le 512 \
-lp 2048 \
-z 256 \
-lr 3e-4 \
-B 0.01 \
-t 500000 \
-wmin 20 \
-wmax 40 \
-i \
-gi \
-tfr \
-n 5 \
--bucket_name=$BUCKET_NAME \
--tpu_name=$TPU_NAME



This the language model
"https://tfhub.dev/google/universal-sentence-encoder/4"

https://stackoverflow.com/questions/60578801/how-to-load-tf-hub-model-from-local-system





python3 train_lfp.py \
IMB002_lang_full_enc \
--bulk_datasets unity/envHz25 \
--train_datasets unity/diverse \
--test_datasets unity/diverse_test \
-tfr \
-s GCS \
-d TPU \
-b 32 \
-la 2048 \
-le 512 \
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
-enc_all


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



python3 augment_language_labelled_data.py --teleop_datasets unity/diverse --bucket_name lfp_europe_west4_a --video_datasets Unity/contrastive_vids



python3 train_lfp.py \
sft_IMB00_lang_lim_enc_v3 \
--bulk_datasets unity/envHz25 unity/augmented_diverse_new \
--train_datasets unity/diverse unity/diverse_new \
--test_datasets unity/diverse_test \
-tfr \
-s GCS \
-d TPU \
-b 32 \
-la 2048 \
-le 512 \
-lp 2048 \
-z 256 \
-lr 3e-4 \
-B 0.0 \
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
--bulk_split 8


-enc_all \
--init_from sft_IMB001_lang_full_enc_v2


python3 train_lfp.py IMB002_lang_full_enc_bigCNNv2 --bulk_datasets unity/envHz25 --train_datasets unity/diverse --test_datasets unity/diverse_test -tfr -s GCS -d TPU -b 32 -la 2048 -le 512 -lp 2048 -z 256 -lr 3e-4 -B 0.02 -n 5 -t 1000000 -wmin 25 -wmax 50 -i -gi -lang --bucket_name=$BUCKET_NAME --tpu_name=$TPU_NAME --standard_split 16 --lang_split 8 --bulk_split 8 -enc_all --init_from IMB002_lang_full_enc_bigCNN