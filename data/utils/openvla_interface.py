from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch

import numpy as np
import os
import tensorflow as tf
import time

from collections import deque
from typing import Optional
from transforms3d.euler import euler2axangle


class OpenVLAModel:
    def __init__(self, policy_setup):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            "openvla/openvla-7b",
            # attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)
        self.policy_setup = policy_setup

    def step(
        self,
        image,
        instruction,
    ):
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        with torch.no_grad():
            inputs = self.processor(prompt, image).to(self.device, dtype=torch.bfloat16)
            if self.policy_setup == "widowx_bridge":
                action = self.vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
            elif self.policy_setup == "google_robot":
                action = self.vla.predict_action(**inputs, unnorm_key="fractal20220817_data", do_sample=False)
            else:
                raise ValueError(f"Policy setup {self.policy_setup} not supported")

        return action


class OpenVLAInferenceWrapper:
    def __init__(
        self,
        model,
        policy_setup: str = "libero",
        horizon: int = 1,
        image_size: int = 256,
        crop: bool = False,
    ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.model = model
        self.policy_setup = policy_setup
        self.image_size = image_size
        self.horizon = horizon

        if policy_setup == 'google_robot':
            self.sticky_gripper_num_repeat = 15
            dataset = "fractal20220817_data"
        elif policy_setup == 'widowx_bridge':
            self.sticky_gripper_num_repeat = 1
            dataset = "bridge_dataset"

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        # self.gripper_is_closed = False
        self.previous_gripper_action = None
        self.close_gripper_num = 0

        self.task = None
        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        self.num_image_history = 0
        self.crop = crop
        self.action_scale = 1.0
        self.late_close_gripper = 2

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        if self.crop:
            scale = 0.9
            offset = int((1 - scale) / 2 * self.image_size + 0.5)
            target_size = int(scale * self.image_size + 0.5)
            image = tf.image.crop_to_bounding_box(image, offset, offset, target_size, target_size)
        image = tf.image.resize(
            image,
            size=(self.image_size, self.image_size),
            method="lanczos3",
            antialias=True,
        )
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

    def reset(self, task_description: str) -> None:

        self.task_description = task_description
        self.image_history.clear()
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        # self.gripper_is_closed = False
        self.previous_gripper_action = None

        self.episode_step = 0

    def step(self, image: np.ndarray, task_description: Optional[str] = None, *args, **kwargs) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
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
        image = Image.fromarray(image)

        # get actions, shape = batch_size * pred_action_horizon * action_dim
        start = time.time()
        raw_actions = self.model.step(
            image,
            task_description,
        )
        end = time.time()

        raw_action, env_action = self.transform_action(raw_actions)

        
        self.episode_step += 1

        return raw_action, env_action, image, (self.task_description, self.task), (end - start)


    def transform_action(self, raw_actions):
        raw_action = {
            "world_vector": np.array(raw_actions[:3]),
            "rotation_delta": np.array(raw_actions[3:6]),
            "open_gripper": np.array(
                raw_actions[6:7]
            ),  # range [0, 1]; 1 = open; 0 = close
        }
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        action_rotation_delta = np.asarray(
            raw_action["rotation_delta"], dtype=np.float64
        )
        roll, pitch, yaw = action_rotation_delta
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle
        action["rot_axangle"] = action_rotation_axangle * self.action_scale

        if self.policy_setup == "google_robot":
            current_gripper_action = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
            # current_gripper_action = raw_action["open_gripper"]
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
            else:
                relative_gripper_action = (
                    self.previous_gripper_action - current_gripper_action
                )
            self.previous_gripper_action = current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and (not self.sticky_action_is_on):
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
            # print(f'action gripper: {action["gripper"]}')

        elif self.policy_setup == "widowx_bridge":
            relative_gripper_action = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
            if relative_gripper_action[0] > 0:
                self.close_gripper_num += 1
            else:
                self.close_gripper_num = 0

            if self.close_gripper_num >= self.late_close_gripper:
                relative_gripper_action[0] = 1
            else:
                relative_gripper_action[0] = -1

            action["gripper"] = relative_gripper_action

        action["terminate_episode"] = np.array([0.0])

        return raw_action, action