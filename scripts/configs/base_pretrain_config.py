from copy import deepcopy
import imp
import os

from ml_collections import ConfigDict, FieldReference
from ml_collections.config_dict import placeholder

get_base_config = imp.load_source(
    "config", os.path.join(os.path.dirname(__file__), "config.py")
).get_config

from octo.data.utils.text_processing import HFTokenizer
from octo.model.components.action_heads import DiffusionActionHead
from octo.model.components.tokenizers import ImageTokenizer, LanguageTokenizer
from octo.model.components.vit_encoders import SmallStem16
from octo.utils.spec import ModuleSpec
from octo.utils.train_utils import hf_weights_loader


def update_config(config, **kwargs):
    updates = ConfigDict(kwargs)
    new_config = deepcopy(config)
    new_config.update(updates)
    return new_config


def get_config(config_string=None):
    model_size, dataset = config_string.split(',')
    config = get_base_config(model_size)

    action_dim = FieldReference(7)
    action_horizon = FieldReference(4)

    # We augment differently for the primary and wrist cameras
    primary_augment_kwargs = dict(
        random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
        random_brightness=[0.1],
        random_contrast=[0.9, 1.1],
        random_saturation=[0.9, 1.1],
        random_hue=[0.05],
        augment_order=[
            "random_resized_crop",
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    )
    wrist_augment_kwargs = dict(
        random_brightness=[0.1],
        random_contrast=[0.9, 1.1],
        random_saturation=[0.9, 1.1],
        random_hue=[0.05],
        augment_order=[
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    )

    # ML-collections complains if the type of an existing field changes
    # so we delete and re-add the field

    del config["dataset_kwargs"]["frame_transform_kwargs"]["resize_size"]
    del config["dataset_kwargs"]["frame_transform_kwargs"]["image_augment_kwargs"]

    config["dataset_kwargs"]["frame_transform_kwargs"]["resize_size"] = {
        "primary": (256, 256),  # workspace camera is at 256x256
        "wrist": (128, 128),  # wrist camera is at 128x128
    }
    config["dataset_kwargs"]["frame_transform_kwargs"]["image_augment_kwargs"] = {
        "primary": primary_augment_kwargs,
        "wrist": wrist_augment_kwargs,
    }

    if dataset == 'metaworld':
        del config["dataset_kwargs"]["oxe_kwargs"]
        dataset_kwargs=dict(
            dataset_kwargs={
                "name": "metaworld_rlds/metaworld_50_demos_per_task",
                "data_dir": "/user/hypervla/dataset",
                "image_obs_keys": {"primary": "image", "wrist": None},
                # "proprio_obs_key": "proprio",
                "language_key": "language_instruction",
                "action_proprio_normalization_type": "normal",
                # We want to avoid normalizing the gripper
                "action_normalization_mask": [True, True, True, False],
                "standardize_fn": "octo.data.oxe.oxe_standardization_transforms:metaworld_dataset_transform",
                # If the default data loading speed is too slow, try these:
                "num_parallel_reads": 16,  # for reading from disk / GCS
                "num_parallel_calls": 16,  # for initial dataset construction
            },
            traj_transform_kwargs=dict(
                action_horizon=action_horizon,
                max_action_dim=4,
                task_augment_strategy="delete_task_conditioning", # TODO: add rephrase?
                # task_augment_kwargs=dict(
                #     paraphrases_repo="rail-berkeley/OXE_paraphrases",
                #     paraphrases_filename="paraphrases_oxe.pkl",
                #     rephrase_prob=0.5,
                # ),
                task_augment_kwargs=dict(
                    keep_image_prob=0.0,
                )
            ),
            batch_size=32,
            shuffle_buffer_size=10000,
        )
    elif dataset == 'google_robot':
        del config["dataset_kwargs"]["oxe_kwargs"]
        dataset_kwargs=dict(
            dataset_kwargs={
                "name": "google_robot",
                "data_dir": "/user/hypervla/dataset",
                "image_obs_keys": {"primary": "image", "wrist": None},
                # "proprio_obs_key": "proprio",
                "language_key": "language_instruction",
                "action_proprio_normalization_type": "normal",
                # We want to avoid normalizing the gripper
                "action_normalization_mask": [True, True, True, True, True, True, False],
                "standardize_fn": "octo.data.oxe.oxe_standardization_transforms:rt1_dataset_transform",
                # If the default data loading speed is too slow, try these:
                "num_parallel_reads": 16,  # for reading from disk / GCS
                "num_parallel_calls": 16,  # for initial dataset construction
                "filter_single_task": placeholder(str),
            },
            traj_transform_kwargs=dict(
                action_horizon=action_horizon,
                max_action_dim=7,
                # task_augment_strategy="delete_and_rephrase",
                # task_augment_kwargs=dict(
                #     paraphrases_repo="rail-berkeley/OXE_paraphrases",
                #     paraphrases_filename="paraphrases_oxe.pkl",
                #     rephrase_prob=0.5,
                #     keep_image_prob=0.0,
                # ),
                task_augment_strategy="delete_task_conditioning",
                task_augment_kwargs=dict(
                    keep_image_prob=0.0,
                ),
                # max_action=5,
            ),
            batch_size=32,
            shuffle_buffer_size=10000,
        )
    else:
        dataset_kwargs=dict(
            oxe_kwargs=dict(
                data_mix="oxe_magic_soup",
                # data_dir="gs://rail-orca-central2/resize_256_256",
                data_dir="/user/hypervla/dataset/oxe",
                load_camera_views=("primary", ),
                load_depth=False,
                force_recompute_dataset_statistics=False,
            ),
            traj_transform_kwargs=dict(
                action_horizon=action_horizon,
                max_action_dim=action_dim,
                task_augment_strategy="delete_and_rephrase",
                task_augment_kwargs=dict(
                    paraphrases_repo="rail-berkeley/OXE_paraphrases",
                    paraphrases_filename="paraphrases_oxe.pkl",
                    rephrase_prob=0.5,
                    keep_image_prob=0.0,
                ),
                # task_augment_strategy="delete_task_conditioning",
                # task_augment_kwargs=dict(
                #     keep_image_prob=0.0,
                # ),
            ),
            batch_size=32,
            shuffle_buffer_size=50000, # origin: 500000
            balance_weights=True,
        )

    if dataset == 'oxe':
        tokenizer_max_length = 32
    else:
        tokenizer_max_length = 16

    config = update_config(
        config,
        num_steps=300000,
        window_size=1, # TODO: check window size here
        optimizer=dict(
            frozen_keys=("*hf_model*",),
        ),
        dataset_kwargs=dataset_kwargs,
        text_processor=ModuleSpec.create(
            HFTokenizer,
            tokenizer_name="t5-base",
            encode_with_model=False,
            tokenizer_kwargs={
                "max_length": tokenizer_max_length,
                "padding": "max_length",
                "truncation": True,
                "return_tensors": "np",
            },
        ),
        pretrained_loaders=(
            ModuleSpec.create(
                hf_weights_loader,
                hf_model="t5-base",
            ),
        ),
        eval_datasets=["bridge_dataset"],
    )

    config["base_net_kwargs"] = dict(
        model_type='vit', # 'cnn', 'vit'
        action_head_type='diffusion', # 'diffusion', 'continuous', 'mix'
        action_horizon=action_horizon,
        action_dim=action_dim,
        cnn_kwargs=dict(
            kernel_sizes=(3, 3, 3, 3),
            strides=(2, 2, 2, 2),
            features=(32, 64, 128, 256),
            padding=(1, 1, 1, 1),
            mlp_hidden_sizes=(32, 32),
        ),
        vit_kwargs=dict(
            encoder_type='SmallStem',
            patch_size=16,
            hidden_dim=64,
            num_layers=4,
            num_heads=4,
            mlp_dim=128,
            dropout_rate=0.0,
            cnn_channels=(32, 96, 192, 384),
            use_language_token=True,
            fine_tune_pretrained_image_encoder=False,
            image_embedding_noise=0.,
        ),
        action_head_kwargs=dict(
            token_per_horizon=False,
            squash_continuous_action=True,
            tanh_scaling_factor=5.,
            clip_target=False,
            max_action=5., # should set to 1 when using bound norm
        ),
    )

    config["auxiliary_loss"] = dict(
        base_weight_decay=0.0,
    )

    return config
