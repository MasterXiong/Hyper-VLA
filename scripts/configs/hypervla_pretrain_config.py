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
    learnable_norm = FieldReference(True)
    use_initial_image = FieldReference(False)

    config["model"]["observation_tokenizers"] = {
        "primary": ModuleSpec.create(
            ImageTokenizer,
            obs_stack_keys=["image_primary"],
            task_stack_keys=["image_primary"],
            encoder=ModuleSpec.create(
                SmallStem16, 
                learnable_norm=learnable_norm
            ),
        ),
        # "wrist": ModuleSpec.create(
        #     ImageTokenizer,
        #     obs_stack_keys=["image_wrist"],
        #     task_stack_keys=["image_wrist"],
        #     encoder=ModuleSpec.create(SmallStem16),
        # ),
    }
    # config["model"]["task_tokenizers"] = {
    #     "language": ModuleSpec.create(
    #         LanguageTokenizer,
    #         encoder="t5-base",
    #         finetune_encoder=False,
    #     ),
    # }
    config["model"]["repeat_task_tokens"] = True
    config["model"]["readouts"] = {"action": 1}
    # config["model"]["heads"]["action"] = ModuleSpec.create(
    #     DiffusionActionHead,
    #     readout_key="readout_action",
    #     use_map=False,
    #     action_horizon=action_horizon,
    #     action_dim=action_dim,
    #     n_diffusion_samples=1,
    #     dropout_rate=0.0,
    # )
    config["model"]["transformer_kwargs"]["learnable_norm"] = learnable_norm
    # config["model"]["max_horizon"] = min(history_horizon, 10)
    config["model"]["max_horizon"] = 10
    config["model"]["use_pretrained_image_tokenizer"] = False

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
    # wrist_augment_kwargs = dict(
    #     random_brightness=[0.1],
    #     random_contrast=[0.9, 1.1],
    #     random_saturation=[0.9, 1.1],
    #     random_hue=[0.05],
    #     augment_order=[
    #         "random_brightness",
    #         "random_contrast",
    #         "random_saturation",
    #         "random_hue",
    #     ],
    # )

    # ML-collections complains if the type of an existing field changes
    # so we delete and re-add the field

    del config["dataset_kwargs"]["frame_transform_kwargs"]["resize_size"]
    del config["dataset_kwargs"]["frame_transform_kwargs"]["image_augment_kwargs"]

    config["dataset_kwargs"]["frame_transform_kwargs"]["resize_size"] = {
        "primary": (256, 256),  # workspace camera is at 256x256
        # "wrist": (128, 128),  # wrist camera is at 128x128
    }
    config["dataset_kwargs"]["frame_transform_kwargs"]["image_augment_kwargs"] = {
        "primary": primary_augment_kwargs,
        # "wrist": wrist_augment_kwargs,
    }
    config["dataset_kwargs"]["frame_transform_kwargs"]["num_parallel_calls"] = 8

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
                # task_augment_strategy="delete_task_conditioning", 
                # task_augment_kwargs=dict(
                #     keep_image_prob=0.0,
                # )
                task_augment_strategy="delete_and_rephrase", 
                task_augment_kwargs=dict(
                    paraphrases_repo="rail-berkeley/OXE_paraphrases",
                    paraphrases_filename="paraphrases_oxe.pkl",
                    rephrase_prob=0.5,
                    keep_image_prob=0.0,
                ),
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
                # "image_obs_keys": {"primary": "image", "wrist": None},
                "image_obs_keys": {"primary": "image"},
                # "proprio_obs_key": "proprio",
                "language_key": "language_instruction",
                "action_proprio_normalization_type": "normal",
                # We want to avoid normalizing the gripper
                "action_normalization_mask": [True, True, True, True, True, True, False],
                "standardize_fn": "octo.data.oxe.oxe_standardization_transforms:rt1_dataset_transform",
                # If the default data loading speed is too slow, try these:
                "num_parallel_reads": 8,  # for reading from disk / GCS
                "num_parallel_calls": 8,  # for initial dataset construction
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
                num_parallel_calls=8,
                skip_unlabeled=False,
            ),
            frame_transform_kwargs=dict(
                apply_image_augmentation=True,
                image_aug_style="octo",
            ),
            batch_size=32,
            shuffle_buffer_size=10000,
        )
    elif dataset == 'libero':
        del config["dataset_kwargs"]["oxe_kwargs"]
        dataset_kwargs=dict(
            dataset_kwargs={
                "name": "libero_rlds/libero_10_all_demos",
                "data_dir": "/user/hypervla/dataset",
                "image_obs_keys": {"primary": "image"},
                "language_key": "language_instruction",
                "action_proprio_normalization_type": "normal",
                # We want to avoid normalizing the gripper
                "action_normalization_mask": [True, True, True, True, True, True, False],
                "standardize_fn": "octo.data.oxe.oxe_standardization_transforms:libero_dataset_transform",
                "num_parallel_reads": 32,
                "num_parallel_calls": 32,
                "add_initial_image": use_initial_image,
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
                num_parallel_calls=32,
                skip_unlabeled=False,
            ),
            frame_transform_kwargs=dict(
                apply_image_augmentation=True,
                image_aug_style="octo",
            ),
            batch_size=256,
            shuffle_buffer_size=20000,
        )
    else:
        if dataset == "oxe":
            image_aug_style = "octo"
        elif dataset == "oxe_pad":
            image_aug_style = "rtx"
        dataset_kwargs=dict(
            oxe_kwargs=dict(
                data_mix="oxe_magic_soup",
                # data_dir="gs://rail-orca-central2/resize_256_256",
                data_dir=f"/user/hypervla/dataset/{dataset}",
                load_camera_views=("primary", ),
                load_depth=False,
                force_recompute_dataset_statistics=False,
                action_proprio_normalization_type="normal",
                add_initial_image=use_initial_image,
            ),
            traj_transform_kwargs=dict(
                action_horizon=action_horizon,
                max_action_dim=action_dim,
                task_augment_strategy="delete_and_rephrase",
                task_augment_kwargs=dict(
                    paraphrases_repo="rail-berkeley/OXE_paraphrases",
                    paraphrases_filename="paraphrases_oxe.pkl",
                    rephrase_prob=1.0,
                    keep_image_prob=0.0,
                ),
                # task_augment_strategy="delete_task_conditioning",
                # task_augment_kwargs=dict(
                #     keep_image_prob=0.0,
                # ),
                skip_unlabeled=False,
            ),
            frame_transform_kwargs=dict(
                apply_image_augmentation=True,
                image_aug_style=image_aug_style,
            ),
            batch_size=32,
            shuffle_buffer_size=50000, # origin: 500000
            balance_weights=True,
            random_initial_image=False,
            initial_image_range=0,
        )

    tokenizer_max_length = 32

    config = update_config(
        config,
        num_steps=300000,
        optimizer=dict(
            frozen_keys=("*hf_model*",),
            weight_decay_strategy='v1',
            base_weight_decay=0.0,
            base_learning_rate=dict(
                name="rsqrt",
                init_value=0.0,
                peak_value=3e-5,
                warmup_steps=2000,
                timescale=10000,
            ),
            grad_accumulation_steps=1,
        ),
        dataset_kwargs=dataset_kwargs,
        text_processor=ModuleSpec.create(
            HFTokenizer,
            tokenizer_name="t5-base",
            encode_with_model=False,
            tokenizer_kwargs={
                "max_length": 32,
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
        save_param_EMA=False,
        EMA_start_step=5000,
        pretrained_checkpoint_path=placeholder(str),
        pretrained_checkpoint_step=placeholder(int),
    )

    config["hypernet_kwargs"] = dict(
        encoder_type='transformer', # "transformer", "mlp"
        context_embedding_dim=128,
        context_encoder_kwargs=dict(
            num_layers=1,
            mlp_dim=256,
            num_attention_heads=4,
            dropout_rate=0.0,
            attention_dropout_rate=0.0,
            add_position_embedding=False,
        ),
        attend_to_padding=False,
        task_attend_to_layer=False,
        embedding_dropout_rate=0.0,
        scale_context_embedding=False,
        one_hot_context=False,
        output_head_bias=True,
        generation_strategy='full',
        shared_modules=tuple(),
        bias_init_with_pretrained_image_tokenizer=False,
        include_goal_image=False,
        use_initial_image=use_initial_image,
        use_all_image_tokens=False,
        share_TF_output_head=False,
        share_CNN=False,
        init_strategy=0,
        share_all_params=False,
        share_layer_index=False,
        image_dropout=0.0,
    )

    config["base_net_kwargs"] = dict(
        model_type='cnn', # 'cnn', 'vit', 'octo'
        action_head_type='diffusion', # 'diffusion', 'continuous', 'mix', 'discrete'
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
            encoder_type='SmallStem', # 'SmallStem', 'EfficientNet'
            patch_size=16,
            hidden_dim=64,
            num_layers=4,
            num_heads=4,
            mlp_dim=128,
            dropout_rate=0.0,
            cnn_channels=(32, 96, 192, 384),
            use_language_token=False,
            fine_tune_pretrained_image_encoder=False,
            image_embedding_noise=0.,
            use_differential_transformer=False,
            return_attention_map=False,
            add_positional_embedding=True,
            include_class_token=False,
        ),
        action_head_kwargs=dict(
            token_per_horizon=False,
            squash_continuous_action=True,
            tanh_scaling_factor=5.,
            clip_target=False,
            max_action=5., # should set to 1 when using bound norm
            hidden_dims=tuple(),
            discrete_token_type="action_dim_and_action_horizon",
            # duffision head
            num_blocks=3,
            hidden_dim=256,
            diffusion_dropout_rate=0.0,
            loss_type="mse",
        ),
    )

    config["auxiliary_loss"] = dict(
        HN_regularizer=0.0,
        close_drawer_weight=1.0,
        attention_map_alignment=0.0,
        attention_entropy=0.0,
        rephrase_strategy=placeholder(str),
        rephrase_alignment_coef=1.0,
    )

    return config
