from lfp.model_dvae import LFPNet
from typing import List

import argparse
from pathlib import Path
import logging

import tensorflow as tf
import wandb

import lfp
# from lfp.model_v2 import create_actor, create_encoder, create_planner, LFPNet


class PrepArgs:

    args: argparse.Namespace

    def __init__(self, args_str=""):

        self._parse_args(args_str)

    # forward attribute accesses to args, so this object can be used in place of args
    def __getattr__(self, attr):
        return getattr(self.args, attr)

    def __repr__(self) -> str:
        return repr(self.args)

    def _parse_args(self, args_str):

        parser = argparse.ArgumentParser(description="LFP training arguments")
        parser.add_argument("run_name")
        parser.add_argument(
            "--train_datasets", nargs="+", help="Training dataset names"
        )
        parser.add_argument("--test_datasets", nargs="+", help="Testing dataset names")
        parser.add_argument(
            "-c",
            "--colab",
            default=False,
            action="store_true",
            help="Enable if using colab environment",
        )
        parser.add_argument(
            "-s", "--data_source", default="DRIVE", help="Source of training data"
        )
        parser.add_argument(
            "-tfr",
            "--from_tfrecords",
            default=False,
            action="store_true",
            help="Enable if using tfrecords format",
        )
        parser.add_argument(
            "-d", "--device", default="TPU", help="Hardware device to train on"
        )
        parser.add_argument("-b", "--batch_size", default=512, type=int)
        parser.add_argument("-wmax", "--window_size_max", default=50, type=int)
        parser.add_argument("-wmin", "--window_size_min", default=20, type=int)
        parser.add_argument(
            "-la",
            "--actor_layer_size",
            default=2048,
            type=int,
            help="Layer size of actor, increases size of neural net",
        )
        parser.add_argument(
            "-le",
            "--encoder_layer_size",
            default=512,
            type=int,
            help="Layer size of encoder, increases size of neural net",
        )
        parser.add_argument(
            "-lp",
            "--planner_layer_size",
            default=512,
            type=int,
            help="Layer size of planner, increases size of neural net",
        )
        parser.add_argument(
            "-embd",
            "--img_embedding_size",
            default=64,
            type=int,
            help="Embedding size of features,goal space",
        )
        parser.add_argument(
            "-z",
            "--latent_dim",
            default=256,
            type=int,
            help="Size of the VAE latent space",
        )
        parser.add_argument(
            "-g",
            "--gcbc",
            default=False,
            action="store_true",
            help="Enables GCBC, a simpler model with no encoder/planner",
        )
        parser.add_argument(
            "-n",
            "--num_distribs",
            default=None,
            type=int,
            help="Number of distributions to use in logistic mixture model",
        )
        parser.add_argument(
            "-q",
            "--qbits",
            default=None,
            type=int,
            help="Number of quantisation bits to discrete distributions into. Total quantisations = 2**qbits",
        )
        parser.add_argument("-lr", "--learning_rate", type=float, default=3e-4)
        parser.add_argument("-t", "--train_steps", type=int, default=200000)
        parser.add_argument("-r", "--resume", default=False, action="store_true")
        parser.add_argument("-B", "--beta", type=float, default=0.00003)
        parser.add_argument("-i", "--images", default=False, action="store_true")
        parser.add_argument("--fp16", default=False, action="store_true")
        parser.add_argument("--bucket_name", help="GCS bucket name to stream data from")
        parser.add_argument(
            "--tpu_name", help="GCP TPU name"
        )  # Only used in the script on GCP

        self.args = parser.parse_args(args_str.split())

class PrepDashboard:

    def __init__(self, args: PrepArgs):

        wandb.login()
        wandb.init(project="learning-from-play_v2")


class PrepPaths:

    STORAGE_PATH: Path
    TRAIN_DATA_PATHS: List[Path]
    TEST_DATA_PATHS: List[Path]

    def __init__(self, args: PrepArgs, mount_gdrive=False):
        # todo: pathy doesn't work nicely with windows, pathlib probably won't work nicely with buckets (haven't tried yet)

        if args.data_source == "DRIVE":
            assert args.colab, "Must be using Colab"

            if mount_gdrive:
                from google.colab import drive, auth
                drive.mount('/content/drive')

            print("Reading data from Google Drive")
            self.STORAGE_PATH = Path("/content/drive/My Drive/Robotic Learning")
        elif args.data_source == "GCS":
            raise NotImplementedError()
            # from colab import auth
            # import requests

            # if self.args.colab:
            #     auth.authenticate_user()
            # print("Reading data from Google Cloud Storage")
            # r = requests.get("https://ipinfo.io")
            # region = r.json()["region"]
            # project_id = "learning-from-play-303306"
            # logging.warning(
            #     f"You are accessing GCS data from {region}, make sure this is the same as your bucket {self.args.bucket_name}"
            # )
            # self.STORAGE_PATH = Pathy(f"gs://{args.bucket_name}")
        else:
            raise NotImplementedError()
            # print("Reading data from local filesystem")
            # self.STORAGE_PATH = WORKING_PATH

        print(f"Storage path: {self.STORAGE_PATH}")
        self.TRAIN_DATA_PATHS = [
            self.STORAGE_PATH.joinpath("data", x) for x in args.train_datasets
        ]
        self.TEST_DATA_PATHS = [
            self.STORAGE_PATH.joinpath("data", x) for x in args.test_datasets
        ]


class PrepDevices:

    NUM_DEVICES: int

    device_strategy: tf.distribute.Strategy

    def __init__(self, args: PrepArgs):

        print("Tensorflow version " + tf.__version__)

        if args.device == "TPU":
            try:
                tpu = tf.distribute.cluster_resolver.TPUClusterResolver(
                    tpu=args.tpu_name
                )  # TPU detection
                print("Running on TPU ", tpu.cluster_spec().as_dict()["worker"])
            except ValueError:
                raise BaseException(
                    "ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!"
                )

            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            self.device_strategy = tf.distribute.TPUStrategy(tpu)
            self.NUM_DEVICES = self.device_strategy.num_replicas_in_sync
            print("REPLICAS: ", self.NUM_DEVICES)
            if args.fp16:
                tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
        else:
            physical_devices = tf.config.list_physical_devices()
            if args.device == "GPU":
                if len(physical_devices) >= 4:
                    tf.config.experimental.set_memory_growth(
                        physical_devices[3], enable=True
                    )
                if args.fp16:
                    tf.keras.mixed_precision.set_global_policy("mixed_float16")
            self.device_strategy = tf.distribute.get_strategy()
            self.NUM_DEVICES = 1
            print(physical_devices)


class PrepDataloader:

    GLOBAL_BATCH_SIZE: int

    dataloader: lfp.data.PlayDataloader
    train_dataset: tf.data.Dataset
    valid_dataset: tf.data.Dataset

    def __init__(self, args: PrepArgs, paths: PrepPaths, num_devices: int):

        self.GLOBAL_BATCH_SIZE = args.batch_size * num_devices
        self.dataloader = lfp.data.PlayDataloader(
            include_imgs=args.images,
            batch_size=self.GLOBAL_BATCH_SIZE,
            window_size=args.window_size_max,
            min_window_size=args.window_size_min,
        )

        # Train data
        train_data = self.dataloader.extract(
            paths.TRAIN_DATA_PATHS, from_tfrecords=args.from_tfrecords
        )
        self.train_dataset = self.dataloader.load(train_data)

        # Validation data
        valid_data = self.dataloader.extract(
            paths.TEST_DATA_PATHS, from_tfrecords=args.from_tfrecords
        )
        self.valid_dataset = self.dataloader.load(valid_data)

    def dims_dict(self):
        return {
            "obs_dim": self.dataloader.obs_dim,
            "act_dim": self.dataloader.act_dim,
            "goal_dim": self.dataloader.goal_dim,
        }


class PrepModelDVAE:

    model: tf.keras.Model

    def __init__(
        self,
        args: PrepArgs,
        device_strategy: tf.distribute.Strategy,
        obs_dim: int,
        act_dim: int,
        goal_dim: int,
        debug=False,
    ):
        from lfp.model_dvae import create_actor, create_encoder, create_planner, LFPNet


        with device_strategy.scope():

            actor = create_actor(
                obs_dim=obs_dim,
                act_dim=act_dim,
                goal_dim=goal_dim,
                layer_size=args.actor_layer_size,
            )
            encoder = create_encoder(
                obs_dim=obs_dim,
                act_dim=act_dim,
                layer_size=args.encoder_layer_size,
            )
            # planner = create_planner(
            #     obs_dim=obs_dim,
            #     goal_dim=goal_dim,
            #     layer_size=args.encoder_layer_size,
            #     latent_dim=args.latent_dim,
            # )

            self.model = LFPNet(encoder, None, actor, beta=args.beta)

            optimizer = tf.keras.optimizers.Adam(args.learning_rate)

            self.model.compile(
                optimizer=optimizer,
                loss="mae",
                steps_per_execution=1,
                run_eagerly=debug,
            )

class PrepModel:

    model: tf.keras.Model

    def __init__(
        self,
        args: PrepArgs,
        device_strategy: tf.distribute.Strategy,
        obs_dim: int,
        act_dim: int,
        goal_dim: int,
        debug=False,
    ):
        from lfp.model_v2 import create_actor, create_encoder, create_planner, LFPNet

        with device_strategy.scope():

            actor = create_actor(
                obs_dim=obs_dim,
                act_dim=act_dim,
                goal_dim=goal_dim,
                latent_dim=args.latent_dim,
                layer_size=args.actor_layer_size,
            )
            encoder = create_encoder(
                obs_dim=obs_dim,
                act_dim=act_dim,
                latent_dim=args.latent_dim,
                layer_size=args.encoder_layer_size,
            )
            planner = create_planner(
                obs_dim=obs_dim,
                goal_dim=goal_dim,
                layer_size=args.encoder_layer_size,
                latent_dim=args.latent_dim,
            )

            self.model = LFPNet(encoder, None, actor, beta=args.beta)

            optimizer = tf.keras.optimizers.Adam(args.learning_rate)

            self.model.compile(
                optimizer=optimizer,
                loss="mae",
                steps_per_execution=1,
                run_eagerly=debug,
            )


class PrepUtils:

    beta_schedule: lfp.train.BetaScheduler
    checkpoint_callback: tf.keras.callbacks.ModelCheckpoint
    step_logger: tf.keras.callbacks.Callback

    def __init__(self, args: PrepArgs, storage_path: Path):

        self.beta_schedule = lfp.train.BetaScheduler(
            "constant", beta_max=args.beta
        )

        self.checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=storage_path.joinpath("saved_models"),
            monitor="val_total_loss",
            save_best_only=True,
            save_weights_only=False,
            save_freq=1000,
        )

        class stepLogger(tf.keras.callbacks.Callback):
            "A Logger that log average performance per `display` steps."

            def __init__(self):
                self.step = 0

            def on_batch_end(self, batch, logs={}):
                self.step += 1

            def on_test_batch_end(self, batch, logs={}):
                wandb.log(logs, step=self.step)
        
        self.step_logger = stepLogger()


class Preparer:

    args: PrepArgs
    dashboard: PrepDashboard
    paths: PrepPaths
    devices: PrepDevices
    dataloader: PrepDataloader
    model: PrepModel
    utils: PrepUtils

    def __init__(self):
        self.args = None
        self.paths = None
        self.devices = None
        self.dataloader = None
        self.model = None
        self.utils = None

    def set_max_notebook_output_height(self, heightEM):

        from IPython.core.display import display, HTML
        display(HTML(f"<style>div.output_scroll {{ max-height: {heightEM}em; }}</style>"))

    def do_one_epoch(self):

        return self.model.model.fit(
            self.dataloader.train_dataset,
            validation_data=self.dataloader.valid_dataset,
            epochs = 1,
            steps_per_epoch = 20,
            validation_steps=1,
        )