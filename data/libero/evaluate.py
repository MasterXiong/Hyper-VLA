import argparse
import numpy as np
import os
import tensorflow as tf
import json
import pickle
import h5py
import jax.numpy as jnp

from libero.libero import get_libero_path
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

from data.utils.hypervla_interface import InferenceWrapper
from data.utils.language_tokenizer import *
from hypervla.model import HyperVLA

import mediapy


# prevent a single jax process from taking up all the GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
gpus = tf.config.list_physical_devices("GPU")
if len(gpus) > 0:
    # prevent a single tf process from taking up all the GPU memory
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=3072)],
    )

def load_model(base_model, input_rng=0, step=None, action_ensemble=False, crop=False, image_horizon=1, action_horizon=4, image_size=224):
    model = InferenceWrapper(
        model=base_model, 
        policy_setup="libero", 
        init_rng=input_rng, 
        action_ensemble=action_ensemble, 
        horizon=image_horizon,
        pred_action_horizon=action_horizon,
        image_size=image_size,
        crop=crop,
    )
    return model


def evaluate(base_model, model_path, task_suite_name, seed=0, checkpoint_step=None, split='train', save_video=False, env_num=20, action_ensemble=False, flipping=False, crop=False, image_horizon=2, recompute=False, EMA_coefficient=None):

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

    if os.path.exists(f'{eval_path}/{save_file_name}.json') and not recompute:
        with open(f'{eval_path}/{save_file_name}.json', 'r') as f:
            all_tasks_success_rate = json.load(f)
    else:
        all_tasks_success_rate = dict()

    with open(f'{model_path}/config.json', 'r') as f:
        train_config = json.load(f)

    action_horizon = train_config["base_net_kwargs"]["action_horizon"]
    # load language tokenizers
    tokenizer, token_embedding_model, t5_params = get_language_tokenizer(train_config["text_processor"]["kwargs"])
    image_size = train_config["dataset_kwargs"]["frame_transform_kwargs"]["resize_size"]["primary"][0]
    if train_config["hypernet_kwargs"].get("use_initial_image", False):
        from transformers import FlaxDinov2Model
        pretrained_image_encoder = FlaxDinov2Model.from_pretrained("facebook/dinov2-base")
        pretrained_params = pretrained_image_encoder.params

    model = load_model(
        base_model, 
        seed, 
        step=checkpoint_step, 
        action_ensemble=action_ensemble, 
        action_horizon=action_horizon,
        image_horizon=image_horizon,
        crop=crop,
        image_size=image_size,
    )

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()

    if task_suite_name == 'libero_90':
        all_task_names = [task.name for task in task_suite.tasks]
        with open('octo/domains/LIBERO/task_split.pkl', 'rb') as f:
            train_tasks, test_tasks = pickle.load(f)
        if 'train' in split:
            tasks = train_tasks
            tasks = [all_task_names.index(task_name[:-10]) for task_name in tasks]
        elif split == 'test':
            tasks = test_tasks
            tasks = [all_task_names.index(task_name[:-10]) for task_name in tasks]
        elif split == 'single_task':
            # TODO: hardcode task name parsing for now
            tasks = [all_task_names.index(model_path.split('/')[2])]
    else:
        tasks = list(range(task_suite.get_num_tasks()))

    @jax.jit
    def DINO_encode_image(raw_images):
        raw_images = raw_images / 255.0
        DINO_image_mean = jnp.array([0.485, 0.456, 0.406])
        DINO_image_std = jnp.array([0.229, 0.224, 0.225])
        raw_images = (raw_images - DINO_image_mean[None, None, None]) / DINO_image_std[None, None, None]
        raw_images = raw_images.transpose(0, 3, 1, 2)
        DINO_outputs = pretrained_image_encoder(pixel_values=raw_images, output_attentions=True)
        return DINO_outputs

    # with open('octo/domains/LIBERO/task_demo_length.pkl', 'rb') as f:
    #     task_demo_length = pickle.load(f)

    if save_video:
        total_runs = 10
    else:
        total_runs = 50

    for task_id in tasks:

        # retrieve a specific task
        task = task_suite.get_task(task_id)
        task_name = task.name
        instruction = task.language
        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

        if not recompute and not save_video:
            if task_name in all_tasks_success_rate and len(all_tasks_success_rate[task_name][1]) >= total_runs:
                continue

        video_path = eval_path.replace("eval_results", "eval_results/video") + f"/{task_name}"
        os.makedirs(video_path, exist_ok=True)

        # approach 1: single process
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": 256,
            "camera_widths": 256
        }
        env = OffScreenRenderEnv(**env_args)
        env.seed(0)
        env.reset()
        init_states = task_suite.get_task_init_states(task_id) # for benchmarking purpose, we fix a set of initial states

        # reset the model with the task instruction
        tokens = tokenizer.encode(instruction)
        instruction_embedding = token_to_embedding(token_embedding_model, t5_params, tokens)
        instruction_dict = {"language_instruction": tokens}
        instruction_dict["language_instruction"]["token_embedding"] = instruction_embedding

        print (f'===== {task_name} =====')
        success_count = 0
        episode_results = []
        for run in range(total_runs):
            env.reset()
            init_state_id = run
            obs = env.set_init_state(init_states[init_state_id])

            for _ in range(10):  # simulate the physics without any actions
                obs, _, _, _ = env.step(np.zeros(7))

            image = obs['agentview_image'][::-1]  # the simulation image is up side down, need to flip manually
            images = [image]

            if train_config["hypernet_kwargs"].get("use_initial_image", False):
                initial_image = model._resize_image(image)
                DINO_outputs = DINO_encode_image(initial_image[None])
                patch_embeddings = jax.lax.stop_gradient(DINO_outputs.last_hidden_state)
                initial_state = {
                    "image_primary": initial_image[None],
                    "patch_embeddings": patch_embeddings,
                    "pad_mask_dict": {
                        "image_primary": np.ones((1, 1)),
                    },
                }
            else:
                initial_state = None
            model.reset(instruction, instruction_dict, initial_state=initial_state)

            success = False
            for t in range(520):
                # step the model; "raw_action" is raw model action output; "action" is the processed action to be sent into maniskill env
                raw_action, action, resized_images, _, model_step_time = model.step(image)
                obs, reward, done, info = env.step(action)
                # update image observation
                image = obs['agentview_image'][::-1]
                images.append(image)
                if done:
                    success = True
                    break
            if success:
                success_count += 1
            episode_results.append(success)
            print(run+1, success_count, success_count/(run+1)*100)
            if save_video:
                result = 'success' if success else 'fail'
                mediapy.write_video(f'{video_path}/{run + 1}_{result}.mp4', images, fps=10)
        env.close()
        all_tasks_success_rate[task_name] = [success_count / total_runs, episode_results]
        print ({key: all_tasks_success_rate[key][0] for key in all_tasks_success_rate})
        if not save_video:
            with open(f'{eval_path}/{save_file_name}.json', 'w') as f:
                json.dump(all_tasks_success_rate, f)

    full_results = sorted(all_tasks_success_rate.items(), key=lambda x: x[0])
    for x in full_results:
        print (x)
    print (np.mean([x[1][0] for x in full_results]))



if __name__ == '__main__':

    # Add arguments
    parser = argparse.ArgumentParser(description="A simple example of argparse")
    parser.add_argument("--model", choices=["hypervla"], default="hypervla", help="The model used for evaluation")
    parser.add_argument("--model_path", type=str, default='', help="The path of the custom model (only useful for octo-custom?)")
    parser.add_argument("--task_suite_name", type=str, default='libero_90', help="the task suite to evaluate on")
    parser.add_argument("--seeds", type=str, default='0+1+2+3', help="seeds for policy and env")
    parser.add_argument("--step", type=int, default=None, help="checkpoint step to evaluate")
    parser.add_argument("--split", type=str, default='train', help="evaluate on the train or test split")
    parser.add_argument("--save_video", action='store_true', help="save evaluation video or not")
    parser.add_argument("--flipping", action='store_true', help="flip the left and right side of the image or not")
    parser.add_argument("--action_ensemble", action='store_true', help="Use action ensemble or not")
    parser.add_argument("--crop", action='store_true', help="Whether to crop the image or not")
    parser.add_argument("--image_horizon", type=int, default=1, help="The horizon of image history")
    parser.add_argument("--recompute", action='store_true', help="Whether to recompute for existing results")
    parser.add_argument("--EMA", type=float, default=None, help="evaluate with EMA of model parameters during training")
    # Parse the arguments
    args = parser.parse_args()

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
        evaluate(base_model, args.model_path, args.task_suite_name, seed=seed, checkpoint_step=args.step, split=args.split, save_video=args.save_video, flipping=args.flipping, action_ensemble=args.action_ensemble, crop=args.crop, image_horizon=args.image_horizon, recompute=args.recompute)
