
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

export TPU_NAME=lfp1
export BUCKET_NAME=lfp_europe_west4_a
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


python3 train_lfp.py \
UnityB0_01 \
--train_dataset Unity/UR5_train \
--test_dataset Unity/UR5_test \
-tfr \
-s GCS \
-d TPU \
-b 512 \
-la 2048 \
-le 512 \
-lp 2048 \
-z 256 \
-lr 3e-4 \
-B 0.01 \
-n 5 \
--bucket_name=$BUCKET_NAME \
--tpu_name=$TPU_NAME