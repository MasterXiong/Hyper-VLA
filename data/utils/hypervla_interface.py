from collections import deque
from typing import Optional
import os
import time
from PIL import Image
import json
import jax
import jax.numpy as jnp
import numpy as np
from octo.model.octo_model import OctoModel
import tensorflow as tf
from transforms3d.euler import euler2axangle

from simpler_env.utils.action.action_ensemble import ActionEnsembler
from octo.data.utils.data_utils import NormalizationType


class InferenceWrapper:
    def __init__(
        self,
        model: Optional[OctoModel] = None,
        policy_setup: str = "libero",
        horizon: int = 1,
        pred_action_horizon: int = 1,
        exec_horizon: int = 1,
        image_size: int = 256,
        init_rng: int = 0,
        action_ensemble: bool = False,
        crop: bool = False,
        save_attention_map: bool = False,
        padded_resize: bool = False,
    ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.model = model
        self.policy_setup = policy_setup
        self.image_size = image_size
        self.horizon = horizon
        self.pred_action_horizon = pred_action_horizon
        self.exec_horizon = exec_horizon
        self.action_ensemble = action_ensemble
        self.action_ensemble_temp = 0.0
        self.padded_resize = padded_resize
        self.rng = jax.random.PRNGKey(init_rng)
        for _ in range(5):
            # the purpose of this for loop is just to match octo server's inference seeds
            self.rng, _key = jax.random.split(self.rng)  # each shape [2,]

        if policy_setup == 'google_robot':
            self.sticky_gripper_num_repeat = 15
            dataset = "fractal20220817_data"
        elif policy_setup == 'widowx_bridge':
            self.sticky_gripper_num_repeat = 1
            dataset = "bridge_dataset"
        elif policy_setup == "libero":
            dataset = "libero"
        else:
            raise ValueError(f"Unknown policy setup: {policy_setup}")

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        # self.gripper_is_closed = False
        self.previous_gripper_action = None

        self.task = None
        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = ActionEnsembler(self.pred_action_horizon, self.action_ensemble_temp)
        else:
            self.action_ensembler = None
        self.num_image_history = 0
        self.crop = crop
        self.save_attention_map = save_attention_map

        if 'action' in model.dataset_statistics:
            self.unnormalization_statistics = model.dataset_statistics['action']
        else:
            self.unnormalization_statistics = model.dataset_statistics[dataset]['action']

        if "dataset_kwargs" in model.config["dataset_kwargs"]:
            self.normalization_type = model.config["dataset_kwargs"]["dataset_kwargs"]["action_proprio_normalization_type"]
        else:
            for dataset_config in model.config["dataset_kwargs"]["dataset_kwargs_list"]:
                if dataset_config["name"] == dataset:
                    self.normalization_type = dataset_config["action_proprio_normalization_type"]
                    break

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        if self.padded_resize:
            image = tf.image.resize_with_pad(
                image,
                target_height=256,
                target_width=320,
            )
        # for resize, pre-training uses lanczos3
        image = tf.image.resize(
            image,
            size=(self.image_size, self.image_size),
            method="lanczos3",
            antialias=True,
        )
        # for crop and resize, pre-training uses bilinear
        if self.crop:
            scale = np.sqrt(0.9)
            # offset = int((1 - scale) / 2 * self.image_size + 0.5)
            # target_size = int(scale * self.image_size + 0.5)
            # image = tf.image.crop_to_bounding_box(image, offset, offset, target_size, target_size)
            # image = tf.image.resize(
            #     image,
            #     size=(self.image_size, self.image_size),
            #     method="lanczos3",
            #     antialias=True,
            # )
            offset = (1 - scale) / 2
            target_size = scale
            bounding_boxes = np.array([offset, offset, offset + target_size, offset + target_size]).reshape(1, 4)
            image = tf.image.crop_and_resize(image[None], bounding_boxes, tf.range(1), (self.image_size, self.image_size))
            image = image[0]
        image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8).numpy()
        return image

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        # Alternative implementation below; but looks like for real eval, filling the entire buffer at the first step is not necessary
        # if self.num_image_history == 0:
        #     self.image_history.extend([image] * self.horizon)
        # else:
        #     self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def _obtain_image_history_and_mask(self) -> tuple[np.ndarray, np.ndarray]:
        images = np.stack(self.image_history, axis=0)
        horizon = len(self.image_history)
        pad_mask = np.ones(horizon, dtype=np.float64)  # note: this should be of float type, not a bool type
        pad_mask[: horizon - min(horizon, self.num_image_history)] = 0
        # pad_mask = np.ones(self.horizon, dtype=np.float64) # note: this should be of float type, not a bool type
        # pad_mask[:self.horizon - self.num_image_history] = 0
        return images, pad_mask

    def reset(self, task_description: str, instruction_dict, initial_state=None) -> None:
        # generate the base model parameters
        if initial_state is not None:
            self.base_params, self.task, intermediate_states = self.model.create_tasks(instruction_dict=instruction_dict, initial_state=initial_state)
        else:
            self.base_params, self.task, intermediate_states = self.model.create_tasks(instruction_dict=instruction_dict)
        # self.task = jax.tree_map(lambda x: np.repeat(x, env_num, axis=0), self.task)
        self.instruction_dict = instruction_dict

        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        # self.gripper_is_closed = False
        self.previous_gripper_action = None

        self.episode_step = 0

    def step(self, image: np.ndarray, task_description: Optional[str] = None, image_embeddings=None, *args, **kwargs) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
            task_description: Optional[str], task description; if different from previous task description, policy state is reset
        Output:
            raw_action: dict; raw policy action output
            action: dict; processed action to be sent to the maniskill2 environment, with the following keys:
                - 'world_vector': np.ndarray of shape (3,), xyz translation of robot end-effector
                - 'rot_axangle': np.ndarray of shape (3,), axis-angle representation of end-effector rotation
                - 'gripper': np.ndarray of shape (1,), gripper action
                - 'terminate_episode': np.ndarray of shape (1,), 1 if episode should be terminated, 0 otherwise
        """
        # TODO: only support image embedding without history for now
        if task_description is not None:
            if task_description != self.task_description:
                # task description has changed; reset the policy state
                self.reset(task_description)

        assert image.dtype == np.uint8
        image = self._resize_image(image)
        self._add_image_to_history(image)
        images, pad_mask = self._obtain_image_history_and_mask()
        images, pad_mask = images[None], pad_mask[None] # add a batch dimension
        # if self.episode_step < 3:
        #     print (images.shape)

        # we need use a different rng key for each model forward step; this has a large impact on model performance
        self.rng, key = jax.random.split(self.rng)  # each shape [2,]
        # print("octo local rng", self.rng, key)

        # get actions, shape = batch_size * pred_action_horizon * action_dim
        start = time.time()
        raw_actions, intermediate_states = self.model.sample_actions(
            images,
            self.instruction_dict,
            self.task,
            pad_mask, 
            self.base_params,
            rng=key,
            image_embeddings=image_embeddings,
        )
        end = time.time()
        raw_actions = raw_actions[0] # squeeze the batch dimension
        if self.save_attention_map:
            # layer_num (12) * head_num (12) * patch_num (256)
            self.dino_attention_map = intermediate_states['intermediates']['encoder']['DINO_attention_map'][0]
            self.dino_attention_map = np.stack([x[0, :, 0, 1:] for x in self.dino_attention_map])
            # layer_num (4) * head_num (4) * patch_num (256)
            self.head_attention_map = intermediate_states['intermediates']['encoder']['Transformer_0']
            try:
                self.head_attention_map = np.stack([self.head_attention_map[f"encoderblock_{i}"]["MultiHeadDotProductAttention_0"]["attention_weights"][0][0, :, -1, :-1] for i in range(4)])
            except:
                self.head_attention_map = np.stack([self.head_attention_map[f"encoderblock_{i}"]['attention_map'][0][0, :, -1, :-1] for i in range(4)])

        if self.normalization_type == NormalizationType.NORMAL:
            mask = self.unnormalization_statistics.get(
                "mask",
                jnp.ones_like(self.unnormalization_statistics["mean"], dtype=bool),
            )
            raw_actions = raw_actions[..., : len(mask)]
            raw_actions = jnp.where(
                mask,
                (raw_actions * self.unnormalization_statistics["std"])
                + self.unnormalization_statistics["mean"],
                raw_actions,
            )
        elif self.normalization_type == NormalizationType.BOUNDS:
            mask = self.unnormalization_statistics.get(
                "mask", jnp.ones_like(self.unnormalization_statistics["p01"], dtype=bool)
            )
            raw_actions = raw_actions[..., : len(mask)]
            raw_actions = jnp.where(
                mask,
                (raw_actions + 1) * (self.unnormalization_statistics["p99"] - self.unnormalization_statistics["p01"] + 1e-8) / 2 + self.unnormalization_statistics["p01"],
                raw_actions,
            )
        else:
            raise ValueError(f"Unknown normalization type: {self.normalization_type}")

        # raw_actions = norm_raw_actions * self.action_std[None] + self.action_mean[None]
        # # use the original policy output for unnormalized action dimension
        # raw_actions = raw_actions * self.action_normalization_mask + norm_raw_actions * (1. - self.action_normalization_mask)

        # TODO: does not support batch-mode action prediction for now
        assert raw_actions.shape == (self.pred_action_horizon, 7)
        if self.action_ensemble:
            raw_action = self.action_ensembler.ensemble_action(raw_actions)
        else:
            raw_action = np.array(raw_actions[0])
        # raw_action is of shape (action_dim, )

        if self.policy_setup == 'metaworld':
            action = raw_action.copy()
            action[:, -1] = 1 - action[:, -1]
        else:
            # process raw_action to obtain the action to be sent to the maniskill2 environment
            action = {}
            action["world_vector"] = raw_action[:3]
            action_rotation_delta = np.asarray(raw_action[3:6], dtype=np.float64)
            roll, pitch, yaw = action_rotation_delta
            action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
            action_rotation_axangle = action_rotation_ax * action_rotation_angle
            action["rot_axangle"] = action_rotation_axangle

            if self.policy_setup == "google_robot":
                current_gripper_action = raw_action[-1].item()

                if self.previous_gripper_action is None:
                    relative_gripper_action = 0
                else:
                    relative_gripper_action = (
                        self.previous_gripper_action - current_gripper_action
                    )  # google robot 1 = close; -1 = open
                self.previous_gripper_action = current_gripper_action

                if np.abs(relative_gripper_action) > 0.5 and self.sticky_action_is_on is False:
                    self.sticky_action_is_on = True
                    self.sticky_gripper_action = relative_gripper_action

                if self.sticky_action_is_on:
                    self.gripper_action_repeat += 1
                    relative_gripper_action = self.sticky_gripper_action

                if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                    self.sticky_action_is_on = False
                    self.gripper_action_repeat = 0
                    self.sticky_gripper_action = 0.0

                action["gripper"] = relative_gripper_action
            elif self.policy_setup == "widowx_bridge":
                action["gripper"] = (
                    2.0 * (raw_action[-1] > 0.5) - 1.0
                )  # binarize gripper action to 1 (open) and -1 (close)
            elif self.policy_setup == 'libero':
                action["gripper"] = 2 * raw_action[-1] - 1
            action = np.concatenate([action["world_vector"], action["rot_axangle"].astype(np.float32), np.array([action["gripper"]]).astype(np.float32)])

        self.episode_step += 1

        return raw_action, action, image, (self.task_description, self.task), (end - start)
