# Available resources for TFRC

* 5 on-demand Cloud TPU v2-8 device(s) in zone us-central1-f
* 100 preemptible Cloud TPU v2-8 device(s) in zone us-central1-f
* 5 on-demand Cloud TPU v3-8 device(s) in zone europe-west4-a

# Creating TPU + VM

```
ctpu up \
--project=learning-from-play-303306 \
--zone=$TPU_ZONE \
--tf-version=2.4.1 \
--name=$TPU_NAME \
--tpu-size=["v2-8"|"v3-8] \
[--preemptible]
```

# ssh into vm instance

```gcloud compute ssh root@$TPU_NAME --zone=$TPU_ZONE```

# optionally clone the repo if not already there
```
git clone https://github.com/sholtodouglas/learning_from_play

cd learning_from_play
./setup.sh
```

# Run the sample training script for GCS setup

```
python3 notebooks/train_lfp.py \
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
--bucket_name=$BUCKET_NAME
--tpu_name=$TPU_NAME
```

**Note** if the ssh session disconnects it will kill the currently running python process.

To see which python processes are currently running use:

```ps -ef | grep python```