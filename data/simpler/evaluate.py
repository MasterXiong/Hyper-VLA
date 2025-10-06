import argparse
import numpy as np
import os
import tensorflow as tf
import json
import pickle
import jax
import time

import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
import sapien.core as sapien
import jax.numpy as jnp

import mediapy
import gymnasium as gym
import matplotlib.pyplot as plt

from data.utils.hypervla_interface import InferenceWrapper
from data.utils.language_tokenizer import *
try:
    from data.utils.openvla_interface import *
except:
    pass
from hypervla.model import HyperVLA
from hypervla.base_model import BaseModel

# prevent a single jax process from taking up all the GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
gpus = tf.config.list_physical_devices("GPU")
if len(gpus) > 0:
    # prevent a single tf process from taking up all the GPU memory
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=3072)],
    )

RT_1_CHECKPOINTS = {
    "rt_1_x": "rt_1_x_tf_trained_for_002272480_step",
    "rt_1_400k": "rt_1_tf_trained_for_000400120",
    "rt_1_58k": "rt_1_tf_trained_for_000058240",
    "rt_1_1k": "rt_1_tf_trained_for_000001120",
}


def get_rt_1_checkpoint(name, ckpt_dir="/user/hypervla/finetune_saves/rtx"):
    assert name in RT_1_CHECKPOINTS, name
    ckpt_name = RT_1_CHECKPOINTS[name]
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    if not os.path.exists(ckpt_path):
        if name == "rt_1_x":
            os.system(f'gsutil -m cp -r gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/{ckpt_name}.zip {ckpt_dir}')
            os.system(f'unzip {ckpt_dir}/{ckpt_name}.zip -d {ckpt_dir}')
        else:
            os.system(f'gsutil -m cp -r gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/{ckpt_name} {ckpt_dir}')
    return ckpt_path


def load_model(model_name, model_path, base_model, policy_setup, input_rng=0, step=None, action_ensemble=False, action_horizon=4, image_horizon=2, crop=False, save_attention_map=False, image_size=256, padded_resize=False):
    if "rt_1" in model_name:
        from simpler_env.policies.rt1.rt1_model import RT1Inference
        ckpt_path = get_rt_1_checkpoint(model_name)
        model = RT1Inference(saved_model_path=ckpt_path, policy_setup=policy_setup)
    elif "octo" in model_name:
        from data.simpler.octo_model import OctoInference
        if model_path != '':
            from octo.model.octo_model import OctoModel
            temp_model = OctoModel.load_pretrained(model_path, step=step)
            model = OctoInference(model=temp_model, policy_setup=policy_setup, init_rng=input_rng, horizon=image_horizon)
        else:
            model = OctoInference(model_type=model_name, policy_setup=policy_setup, init_rng=input_rng, horizon=image_horizon)
    elif model_name == "openvla":
            tempmodel = OpenVLAModel(policy_setup)
            model = OpenVLAInferenceWrapper(
                model=tempmodel,
                policy_setup=policy_setup, 
                horizon=1,
            )
    elif model_name == 'hypervla':
        model = InferenceWrapper(
            model=base_model, 
            policy_setup=policy_setup, 
            init_rng=input_rng, 
            action_ensemble=action_ensemble, 
            horizon=image_horizon,
            pred_action_horizon=action_horizon,
            image_size=image_size,
            crop=crop,
            save_attention_map=save_attention_map,
            padded_resize=padded_resize,
        )
    elif model_name == 'base_net':
        tempmodel = BaseModel.load_pretrained(model_path, step=step)
        model = InferenceWrapper(
            model=tempmodel, 
            policy_setup=policy_setup, 
            init_rng=input_rng, 
            action_ensemble=action_ensemble, 
            horizon=image_horizon,
            pred_action_horizon=action_horizon,
            image_size=image_size,
            crop=crop,
        )
    return model


def evaluate(model_name, model_path, base_model, tasks, seed=0, checkpoint_step=None, action_ensemble=True, image_horizon=2, save_video=False, save_trajectory=False, recompute=False, crop=False, save_attention_map=False, EMA_coefficient=None):

    previous_policy_setup = ''
    if model_path == '':
        eval_path = f'eval_results/google_robot/{model_name}/{seed}'
    else:
        save_dir = model_path.replace('finetune_saves', 'eval_results')
        eval_path = f'{save_dir}/eval_step_{checkpoint_step}/{seed}'
    os.makedirs(eval_path, exist_ok=True)

    save_file_name = f'success_rate'
    if action_ensemble:
        save_file_name += '_action_ensemble'
    save_file_name += f'_horizon_{image_horizon}'
    if crop:
        save_file_name += '_crop'
    if EMA_coefficient is not None:
        save_file_name += f'_EMA_{EMA_coefficient}'

    if os.path.exists(f'{eval_path}/{save_file_name}.json'):
        with open(f'{eval_path}/{save_file_name}.json', 'r') as f:
            all_tasks_success_rate = json.load(f)
    else:
        all_tasks_success_rate = dict()

    if model_path != '':
        with open(f'{model_path}/config.json', 'r') as f:
            train_config = json.load(f)

    if model_name in ["hypervla", "base_net"]:
        action_horizon = train_config["base_net_kwargs"]["action_horizon"]
        # load language tokenizers
        tokenizer, token_embedding_model, t5_params = get_language_tokenizer(train_config["text_processor"]["kwargs"])
        image_size = train_config["dataset_kwargs"]["frame_transform_kwargs"]["resize_size"]["primary"][0]
        if train_config["dataset_kwargs"]["frame_transform_kwargs"].get("image_aug_style", "octo") == "rtx":
            padded_resize = True
        else:
            padded_resize = False
        if train_config["hypernet_kwargs"].get("use_initial_image", False):
            from transformers import FlaxDinov2Model
            pretrained_image_encoder = FlaxDinov2Model.from_pretrained("facebook/dinov2-base")
            pretrained_params = pretrained_image_encoder.params
    else:
        action_horizon = 4
        image_size = 256
        padded_resize = False

    @jax.jit
    def DINO_encode_image(raw_images):
        raw_images = raw_images / 255.0
        DINO_image_mean = jnp.array([0.485, 0.456, 0.406])
        DINO_image_std = jnp.array([0.229, 0.224, 0.225])
        raw_images = (raw_images - DINO_image_mean[None, None, None]) / DINO_image_std[None, None, None]
        raw_images = raw_images.transpose(0, 3, 1, 2)
        DINO_outputs = pretrained_image_encoder(pixel_values=raw_images, output_attentions=True)
        return DINO_outputs

    for task_name in tasks:

        if not recompute and not save_video:
            if task_name in all_tasks_success_rate and len(all_tasks_success_rate[task_name][1]) == tasks[task_name][1]:
                continue

        video_path = eval_path.replace("eval_results", "eval_results/video") + f"/{task_name}"
        os.makedirs(video_path, exist_ok=True)
        # skip the current task if videos have been generated before
        if not recompute and len(os.listdir(video_path)) >= 10:
            continue

        if "google" in task_name:
            policy_setup = "google_robot"
        else:
            policy_setup = "widowx_bridge"

        # reduce the number of model loading
        if policy_setup != previous_policy_setup:
            model = load_model(
                model_name, 
                model_path, 
                base_model, 
                policy_setup, 
                seed, 
                step=checkpoint_step, 
                action_ensemble=action_ensemble, 
                action_horizon=action_horizon,
                image_horizon=image_horizon,
                crop=crop,
                save_attention_map=save_attention_map,
                image_size=image_size,
                padded_resize=padded_resize,
            )
        previous_policy_setup = policy_setup

        if model_name == "hypervla":
            if train_config["base_net_kwargs"]["vit_kwargs"]["encoder_type"] == "EfficientNet":
                model.image_size = 300

        if 'env' in locals():
            print("Closing existing env")
            env.close()
            del env

        env_name, total_runs, options = tasks[task_name]
        if env_name is not None:
            kwargs = dict()
            kwargs["prepackaged_config"] = True
            env = gym.make(env_name, obs_mode="rgbd", **kwargs)
        else:
            env = simpler_env.make(task_name)

        if save_video or save_attention_map:
            total_runs = 10

        # turned off the denoiser as the colab kernel will crash if it's turned on
        sapien.render_config.rt_use_denoiser = False

        print (f'===== {task_name} =====')
        obs, reset_info = env.reset(seed=seed)

        success_count = 0
        episode_results = []
        for run in range(total_runs):
            if options is not None:
                obs, reset_info = env.reset(options=options[run])
            else:
                obs, reset_info = env.reset()

            instruction = env.get_language_instruction()
            is_final_subtask = env.is_final_subtask()
            print (instruction, is_final_subtask)

            if model_name in ["hypervla", "base_net"]:
                tokens = tokenizer.encode(instruction)
                if os.path.exists(f"{model_path}/eval_instruction_embeddings.pkl"):
                    with open(f"{model_path}/eval_instruction_embeddings.pkl", "rb") as f:
                        all_instruction_embeddings = pickle.load(f)
                    if instruction in all_instruction_embeddings:
                        instruction_embedding = all_instruction_embeddings[instruction][None]
                        print ('use the same instruction embedding as training')
                    else:
                        instruction_embedding = token_to_embedding(token_embedding_model, t5_params, tokens)
                        print ("re-generate instruction embedding, which may not be the same as training")
                else:
                    instruction_embedding = token_to_embedding(token_embedding_model, t5_params, tokens)
                    print ("re-generate instruction embedding, which may not be the same as training")
                instruction_dict = {"language_instruction": tokens}
                instruction_dict["language_instruction"]["token_embedding"] = instruction_embedding
                # with open("reference_instruction.pkl", "wb") as f:
                #     pickle.dump(instruction_dict, f)
                # with open("reference_instruction.pkl", "rb") as f:
                #     reference_instruction_dict = pickle.load(f)
                # assert (reference_instruction_dict["language_instruction"]["token_embedding"] == instruction_dict["language_instruction"]["token_embedding"]).all()

            image = get_image_from_maniskill2_obs_dict(env, obs)  # np.ndarray of shape (H, W, 3), uint8

            if model_name == 'hypervla':
                if train_config["hypernet_kwargs"].get("use_initial_image", False):
                    initial_image = model._resize_image(image)
                    DINO_outputs = DINO_encode_image(initial_image[None])
                    patch_embeddings = jax.lax.stop_gradient(DINO_outputs.last_hidden_state)
                    initial_state = {
                        "image_primary": initial_image,
                        "patch_embeddings": patch_embeddings,
                        "pad_mask_dict": {
                            "image_primary": np.ones((1, 1)),
                        },
                    }
                else:
                    initial_state = None
                model.reset(instruction, instruction_dict, initial_state=initial_state)
            elif model_name == 'base_net':
                model.reset(instruction, instruction_dict)
            else:
                model.reset(instruction)
            # with open('base_params.pkl', "wb") as f:
            #     pickle.dump(model.base_params, f)
            # with open('base_params.pkl', "rb") as f:
            #     reference_base_params = pickle.load(f)
            # equal = jax.tree_map(lambda x, y: (x ==y).all(), reference_base_params, model.base_params)
            # assert all(jax.tree_util.tree_leaves(equal)), "Base params are different"

            images = []
            predicted_terminated, success, truncated = False, False, False
            model_time, inference_time, sim_time = [], [], []
            action_sequence = []
            step_count = 0
            dino_attention_maps, head_attention_maps = [], []
            while not (truncated or success):
                start = time.time()
                # step the model; "raw_action" is raw model action output; "action" is the processed action to be sent into maniskill env
                if model_name in ['hypervla', 'base_net']:
                    raw_action, action, resized_image, _, model_step_time = model.step(image)
                else:
                    if model_name == "openvla":
                        raw_action, action, resized_image, _, model_step_time = model.step(image, instruction) # add an additional batch dimension
                    else:
                        raw_action, action = model.step(image) # add an additional batch dimension
                    action = np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]])
                    resized_image = image
                    model_step_time = 0
                end = time.time()

                if save_attention_map:
                    dino_attention_map = model.dino_attention_map
                    head_attention_map = model.head_attention_map
                    dino_attention_maps.append(dino_attention_map)
                    head_attention_maps.append(head_attention_map)

                # with open(f"reference_image_{step_count}.pkl", "wb") as f:
                #     pickle.dump(resized_image, f)
                # with open(f"reference_image_{step_count}.pkl", "rb") as f:
                #     reference_image = pickle.load(f)
                # assert (reference_image == resized_image).all()

                # # with open(f"reference_action_{step_count}.pkl", "wb") as f:
                # #     pickle.dump(raw_action, f)
                # with open(f"reference_action_{step_count}.pkl", "rb") as f:
                #     reference_action = pickle.load(f)
                # assert (reference_action == raw_action).all()

                inference_time.append(end - start)
                model_time.append(model_step_time)
                images.append(resized_image.squeeze())
                action_sequence.append(raw_action)
                # predicted_terminated = bool(action["terminate_episode"][0] > 0)
                # if predicted_terminated:
                #     if not is_final_subtask:
                #         # advance the environment to the next subtask
                #         predicted_terminated = False
                #         env.advance_to_next_subtask()
                start = time.time()
                obs, reward, success, truncated, info = env.step(action)
                end = time.time()
                sim_time.append(end - start)

                # new_instruction = env.get_language_instruction()
                # if new_instruction != instruction:
                #     # update instruction for long horizon tasks
                #     instruction = new_instruction
                #     print (instruction)
                # is_final_subtask = env.is_final_subtask()
                # update image observation
                image = get_image_from_maniskill2_obs_dict(env, obs)
                step_count += 1
            if success:
                success_count += 1
            episode_results.append(success)
            print(run+1, success_count, success_count/(run+1)*100)
            print (f'avg inference time: {np.mean(inference_time)}, model: {np.mean(model_time)}, sim: {np.mean(sim_time)}')
            result = 'success' if success else 'fail'
            if save_video:
                mediapy.write_video(f'{video_path}/{run + 1}_{result}_{instruction}.mp4', images, fps=10)
            if save_attention_map:
                attention_map_path = eval_path.replace("eval_results", "attention_map")
                if crop:
                    attention_map_path += '_crop'
                if EMA_coefficient is not None:
                    attention_map_path += f'_EMA_{EMA_coefficient}'
                os.makedirs(f"{attention_map_path}/attention_map_head/{task_name}", exist_ok=True)
                with open(f"{attention_map_path}/attention_map_head/{task_name}/{run}_{instruction}_{result}.pkl", "wb") as f:
                    pickle.dump(head_attention_maps, f)
                os.makedirs(f"{attention_map_path}/attention_map_dino/{task_name}", exist_ok=True)
                with open(f"{attention_map_path}/attention_map_dino/{task_name}/{run}_{instruction}_{result}.pkl", "wb") as f:
                    pickle.dump(dino_attention_maps, f)
                os.makedirs(f"{attention_map_path}/image/{task_name}", exist_ok=True)
                with open(f"{attention_map_path}/image/{task_name}/{run}_{instruction}_{result}.pkl", "wb") as f:
                    pickle.dump(images, f)
            if save_trajectory:
                os.makedirs(f"{eval_path}/trajectory/{task_name}", exist_ok=True)
                with open(f"{eval_path}/trajectory/{task_name}/{run}_{instruction}_{result}.pkl", "wb") as f:
                    pickle.dump([images, action_sequence], f)
                # save the action sequences
                # action_sequence = np.stack(action_sequence)
                # plt.figure(figsize=(20, 8))
                # for i in range(3):
                #     plt.subplot(2, 4, i + 1)
                #     plt.plot(action_sequence[:, i])
                #     plt.title(f"world vector {i}")
                # for i in range(3):
                #     plt.subplot(2, 4, i + 5)
                #     plt.plot(action_sequence[:, i + 3])
                #     plt.title(f"rotation {i}")
                # plt.subplot(2, 4, 4)
                # plt.plot(action_sequence[:, -1])
                # plt.title("gripper")
                # plt.tight_layout()
                # plt.savefig(f"{video_path}/{run + 1}_{result}_{instruction}.png")
                # plt.close()
        env.close()
        all_tasks_success_rate[task_name] = [success_count / total_runs, episode_results]
        print ({key: all_tasks_success_rate[key][0] for key in all_tasks_success_rate})
        if not save_video and not save_attention_map:
            with open(f'{eval_path}/{save_file_name}.json', 'w') as f:
                json.dump(all_tasks_success_rate, f)



if __name__ == '__main__':

    # Add arguments
    parser = argparse.ArgumentParser(description="A simple example of argparse")
    parser.add_argument("--model", choices=["octo-base-1.5", "rt_1_x", "rt_1_400k", "openvla", "hypervla", "base_net"], default="hypervla", help="The model used for evaluation")
    parser.add_argument("--model_path", type=str, default='', help="The path of the custom model (only useful for octo-custom?)")
    parser.add_argument("--seeds", type=str, default='0+1+2+3', help="seeds for policy and env")
    parser.add_argument("--step", type=int, default=None, help="checkpoint step to evaluate")
    parser.add_argument("--action_ensemble", action='store_true', help="use action ensemble or not")
    parser.add_argument("--save_video", action='store_true', help="save evaluation video or not")
    parser.add_argument("--save_trajectory", action='store_true', help="save eval trajectory or not")
    parser.add_argument("--recompute", action='store_true', help="whether to overwrite existing eval results")
    parser.add_argument("--window_size", type=int, default=2, help="window size of historical observations")
    parser.add_argument("--crop", action='store_true', help="whether to crop the resized image or not")
    parser.add_argument("--save_attention_map", action='store_true', help="whether to save attention map of DINOv2 or not")
    parser.add_argument("--EMA", type=float, default=None, help="evaluate with EMA of model parameters during training")
    # Parse the arguments
    args = parser.parse_args()

    # define tasks
    move_task_options = [{"obj_init_options": {"episode_id": i}} for i in range(60)]
    tasks = {
        "google_robot_close_top_drawer": (None, 20, None),
        "google_robot_close_middle_drawer": (None, 20, None),
        "google_robot_close_bottom_drawer": (None, 20, None),
        "google_robot_pick_object": (None, 50, None), 
        "google_robot_move_near": (None, len(move_task_options), move_task_options),
        "widowx_spoon_on_towel": (None, 20, None),
        "widowx_carrot_on_plate": (None, 20, None),
        "widowx_stack_cube": (None, 20, None),
        "widowx_put_eggplant_in_basket": (None, 20, None),
    }

    if args.model == 'hypervla':
        base_model = HyperVLA.load_pretrained(args.model_path, step=args.step)
        if args.EMA is not None:
            with open(f"{args.model_path}/{args.step}/EMA_params.pkl", "rb") as f:
                EMA_params = pickle.load(f)
            base_model = base_model.replace(params=EMA_params[f"EMA_{args.EMA}"])
            del EMA_params
    else:
        base_model = None

    seeds = [eval(seed) for seed in args.seeds.split('+')]
    for seed in seeds:
        evaluate(args.model, args.model_path, base_model, tasks, seed=seed, checkpoint_step=args.step, action_ensemble=args.action_ensemble, save_video=args.save_video, save_trajectory=args.save_trajectory, recompute=args.recompute, image_horizon=args.window_size, crop=args.crop, save_attention_map=args.save_attention_map, EMA_coefficient=args.EMA)
