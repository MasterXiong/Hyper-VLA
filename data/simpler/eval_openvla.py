from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch

import argparse
import numpy as np
import os
import tensorflow as tf
import json
import pickle
import time

import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
import sapien.core as sapien

import mediapy
import gymnasium as gym
import matplotlib.pyplot as plt

from data.utils.multi_env_interface import OpenVLAInferenceWrapper


class OpenVLAModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            "openvla/openvla-7b",
            # attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)

    def step(
        self,
        image,
        instruction,
    ):
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        inputs = self.processor(prompt, image).to(self.device, dtype=torch.bfloat16)
        action = self.vla.predict_action(**inputs, unnorm_key="fractal20220817_data", do_sample=False)

        return action



def evaluate_openvla(
    model_name,
    model_path,
    tasks,
    seed=0,
    checkpoint_step=None,
    action_ensemble=False,
    save_video=False,
    save_trajectory=False,
    recompute=False,
    image_horizon=2,
):

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
    elif model_name == "openvla":
        action_horizon = 4
    else:
        action_horizon = 4

    for task_name in tasks:

        if not recompute and not save_video:
            if task_name in all_tasks_success_rate and len(all_tasks_success_rate[task_name][1]) == tasks[task_name][1]:
                continue
        
        video_path = f"{eval_path}/video/{task_name}"
        os.makedirs(video_path, exist_ok=True)
        # skip the current task if videos have been generated before
        if not recompute and len(os.listdir(video_path)) >= 10:
            continue

        if "google" in task_name:
            policy_setup = "google_robot"
        else:
            policy_setup = "widowx_bridge"

        if policy_setup != previous_policy_setup:
            tempmodel = OpenVLAModel()
            model = OpenVLAInferenceWrapper(
                model=tempmodel,
                policy_setup=policy_setup, 
                action_ensemble=action_ensemble, 
                horizon=image_horizon,
                pred_action_horizon=action_horizon,
            )
        previous_policy_setup = policy_setup

        if model_name == "openvla":
            pass

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

        if save_video:
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

            if model_name in ['openvla']:
                pass

            image = get_image_from_maniskill2_obs_dict(env, obs)
            if model_name in ['openvla']:
                model.reset(instruction)
            images = []
            predicted_terminated, success, truncated = False, False, False
            model_time, inference_time, sim_time = [], [], []
            action_sequence = []
            step_count = 0
            while not (truncated or success):
                start = time.time()
                # step the model; "raw_action" is raw model action output; "action" is the processed action to be sent into maniskill env
                if model_name in ['openvla']:
                    raw_action, action, resized_image, _, model_step_time = model.step(image, instruction) # add an additional batch dimension
                    action = np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]])
                    resized_image = image
                    model_step_time = 0
                end = time.time()
                inference_time.append(end - start)
                model_time.append(model_step_time)
                images.append(resized_image)
                action_sequence.append(action)
                start = time.time()
                obs, reward, success, truncated, info = env.step(action)
                end = time.time()
                sim_time.append(end - start)
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
            if save_trajectory:
                os.makedirs(f"{eval_path}/trajectory/{task_name}", exist_ok=True)
                with open(f"{eval_path}/trajectory/{task_name}/{run}_{instruction}_{result}.pkl", "wb") as f:
                    pickle.dump([images, action_sequence], f)
            
        env.close()
        all_tasks_success_rate[task_name] = [success_count / total_runs, episode_results]
        print ({key: all_tasks_success_rate[key][0] for key in all_tasks_success_rate})
        if not save_video:
            with open(f'{eval_path}/{save_file_name}.json', 'w') as f:
                json.dump(all_tasks_success_rate, f)

"""
Does model need to be reset?
History Observation?
"""


if __name__ == "__main__":

    # Add arguments
    parser = argparse.ArgumentParser(description="A simple example of argparse")
    parser.add_argument("--model", default="openvla", help="The model used for evaluation")
    parser.add_argument("--model_path", type=str, default='', help="The path of the custom model (only useful for octo-custom?)")
    parser.add_argument("--seeds", type=str, default='0+1+2+3', help="seeds for policy and env")
    parser.add_argument("--step", type=int, default=None, help="checkpoint step to evaluate")
    parser.add_argument("--action_ensemble", action='store_true', help="use action ensemble or not")
    parser.add_argument("--save_video", action='store_true', help="save evaluation video or not")
    parser.add_argument("--save_trajectory", action='store_true', help="save eval trajectory or not")
    parser.add_argument("--recompute", action='store_true', help="whether to overwrite existing eval results")
    parser.add_argument("--window_size", type=int, default=2, help="window size of historical observations")
    # Parse the arguments
    args = parser.parse_args()

    # define tasks
    move_task_options = [{"obj_init_options": {"episode_id": i}} for i in range(60)]
    tasks = {
        # "google_robot_pick_coke_can": (None, 50, None),
        "google_robot_pick_object": (None, 50, None), 
        "google_robot_move_near": (None, len(move_task_options), move_task_options),
        "google_robot_close_top_drawer": (None, 50, None),
        "google_robot_close_middle_drawer": (None, 50, None),
        "google_robot_close_bottom_drawer": (None, 50, None),
        # "google_robot_open_top_drawer": (None, 20, None),
        # "google_robot_open_middle_drawer": (None, 20, None),
        # "google_robot_open_bottom_drawer": (None, 20, None),
        # "google_robot_place_apple_in_closed_top_drawer": (None, 10, None),
        # "widowx_spoon_on_towel": (None, 20, None),
        # "widowx_carrot_on_plate": (None, 20, None),
        # "widowx_stack_cube": (None, 20, None),
        # "widowx_put_eggplant_in_basket": (None, 20, None),
    }

    seeds = [eval(seed) for seed in args.seeds.split('+')]
    for seed in seeds:
        evaluate_openvla(args.model, args.model_path, tasks, seed=seed, checkpoint_step=args.step, action_ensemble=args.action_ensemble, save_video=args.save_video, save_trajectory=args.save_trajectory, recompute=args.recompute, image_horizon=args.window_size)

    