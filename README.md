# HyperVLA

This is the codebase of the paper [HyperVLA: Efficient Inference in Vision-Language-Action Models via Hypernetworks
](https://arxiv.org/abs/2510.04898), which significantly accelerates VLA inference by generating compact task-specific policy network via hypernetworks.

## Installation
Run `bash docker/simpler_cuda12/build_docker.sh` to install the docker environment

## Start the docker
Run the following command to start the docker
```
docker run --gpus '"device=xxx"' --rm --network host --ipc=host --user $(id -u) -v code_path:/user/hypervla -v dataset_path:/user/hypervla/dataset -v checkpoint_save_path:/user/hypervla/finetune_saves -v evaluation_result_path:/user/hypervla/eval_results -it hypervla_simpler /bin/bash
```

## Train
By default, we train HyperVLA on the Open X-Embodiment (OXE) dataset. Run the following command to train HyperVLA
```
python -m scripts.train \
    --config scripts/configs/hypervla_pretrain_config.py:vit_t,oxe \
    --config.wandb.project=`your_wandb_project_name` \
    --name=`your_run_name` \
    --config.save_dir=`your_save_path` \
    --config.dataset_kwargs.batch_size=256 \
    --config.dataset_kwargs.shuffle_buffer_size=250000 \
    --config.dataset_kwargs.traj_transform_kwargs.skip_unlabeled=True \
    --config.dataset_kwargs.frame_transform_kwargs.resize_size.primary='(224, 224)' \
    --config.save_interval=10000 \
    --config.num_steps=100000 \
    --config.optimizer.weight_decay_strategy=v5 \
    --config.optimizer.weight_decay=0.05 \
    --config.optimizer.base_weight_decay=0.0 \
    --config.optimizer.grad_accumulation_steps=1 \
    --config.hypernet_kwargs.context_embedding_dim=128 \
    --config.hypernet_kwargs.context_encoder_kwargs.num_layers=6 \
    --config.hypernet_kwargs.context_encoder_kwargs.mlp_dim=512 \
    --config.hypernet_kwargs.context_encoder_kwargs.num_attention_heads=4 \
    --config.hypernet_kwargs.scale_context_embedding=True \
    --config.hypernet_kwargs.generation_strategy='block' \
    --config.hypernet_kwargs.attend_to_padding=False \
    --config.hypernet_kwargs.embedding_dropout_rate=0.0 \
    --config.hypernet_kwargs.share_layer_index=True \
    --config.hypernet_kwargs.shared_modules='("image_encoder", )' \
    --config.hypernet_kwargs.use_initial_image=True \
    --config.hypernet_kwargs.share_TF_output_head=False \
    --config.base_net_kwargs.model_type=vit \
    --config.base_net_kwargs.vit_kwargs.encoder_type=DINOv2 \
    --config.base_net_kwargs.vit_kwargs.num_layers=4 \
    --config.base_net_kwargs.vit_kwargs.hidden_dim=64 \
    --config.base_net_kwargs.vit_kwargs.num_heads=4 \
    --config.base_net_kwargs.vit_kwargs.mlp_dim=128 \
    --config.base_net_kwargs.vit_kwargs.dropout_rate=0.0 \
    --config.base_net_kwargs.vit_kwargs.use_differential_transformer=False \
    --config.base_net_kwargs.vit_kwargs.add_positional_embedding=True \
    --config.base_net_kwargs.vit_kwargs.use_language_token=False \
    --config.base_net_kwargs.vit_kwargs.fine_tune_pretrained_image_encoder=True \
    --config.base_net_kwargs.action_head_type=mix \
    --config.base_net_kwargs.action_head_kwargs.clip_target=True \
    --config.base_net_kwargs.action_head_kwargs.squash_continuous_action=True \
    --config.base_net_kwargs.action_head_kwargs.tanh_scaling_factor=5.0 \
    --config.window_size=1 \
    --config.save_param_EMA=True \
    --config.auxiliary_loss.rephrase_strategy=replace \
    --config.seed=2025
```

## Zero-shot evaluation on SIMPLER
```
python -m data.simpler.evaluate --model hypervla --model_path `your_model_path` --step 100000 --action_ensemble --window_size 1 --seeds 0 --crop --EMA 0.999
```

## Few-shot adaptation on LIBERO
To evaluate on LIBERO, you need to first fine-tune HyperVLA on the LIBERO demonstrations. You can reuse the training command for fine-tuning by adding the following configs to load a pretrained model checkpoint: 
```
--config.pretrained_checkpoint_path=`your_pretrained_checkpoint_path` \
--config.pretrained_checkpoint_step=100000
```
After fine-tuning, evaluate with the following command:
```
python -m data.libero.evaluate --model hypervla --model_path `your_model_path` --task_suite_name libero_object --step 10000 --action_ensemble --seeds 0 --crop --EMA 0.999
```

## Acknowledgements
This project is developed based on the [Octo codebase](https://github.com/octo-models/octo). We thank the Octo team for open-sourcing this great work. 
