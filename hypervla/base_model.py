import json
import logging
from typing import Any, Optional

import flax
from flax import struct
from flax import linen as nn
from flax.training import orbax_utils
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint
import tensorflow as tf

from octo.data.utils.text_processing import TextProcessor
from octo.utils.spec import ModuleSpec
from octo.utils.typing import Config, Data, Params, PRNGKey

from hypervla.components.base_network import *
from hypervla.utils import *


@struct.dataclass
class BaseModel:

    base_net: nn.Module = struct.field(pytree_node=False)
    text_processor: TextProcessor = struct.field(pytree_node=False)
    config: Config = struct.field(pytree_node=False)
    params: Params
    example_batch: Data
    dataset_statistics: Optional[Data]

    def create_tasks(
        self, 
        goals: Optional[Data] = None, 
        instruction_dict: dict = None,
    ):
        """Creates tasks dict from goals and texts.

        Args:
            goals: if not None, dict of arrays with shape (batch_size, *)
            texts: if not None, list of texts of length batch_size

        Omit images to run the language-conditioned model, and omit texts to run the
        goal-conditioned model.
        """
        return self.params, None

    @jax.jit
    def sample_actions(
        self,
        images,
        instruction_dict,
        task,
        timestep_pad_mask,
        base_params,
        train: bool = False,
        rng: Optional[PRNGKey] = None,
        save_attention_map = False,
        # unnormalization_statistics: Optional[Data] = None,
        # normalization_type: NormalizationType = NormalizationType.NORMAL,
        # timestep_pad_mask: Optional[ArrayLike] = None,
        # train: bool = False,
        # argmax: bool = False,
        # sample_shape: Tuple[int, ...] = (),
        # rng: Optional[PRNGKey] = None,
        # temperature: float = 1.0,
    ):
        """Samples actions from the model. See `action_heads.py` for more info.

        Args:
            observations: dictionary of arrays of shape (batch_size, window_size, *)
            tasks: dict of tasks of shape (batch_size, *)
            unnormalization_statistics: dict of statistics for unnormalizing actions (must contain "mean",
                "std", and optionally "mask")
            normalization_type: type of normalization applied to the actions
            timestep_pad_mask: (batch_size, window_size) Boolean mask that is False when the timestep corresponds to padding
            train: whether to run in train mode
            ...see `action_heads.py` for the rest of the kwargs.
        Returns:
            actions: (*sample_shape, batch_size, action_horizon, action_dim)
        """
        action = self.base_net.apply({'params': base_params}, images, instruction_dict["language_instruction"]["token_embedding"], timestep_pad_mask, rng=rng, train=train, method=BaseNetwork.predict_action)
        return action, None

    @classmethod
    def load_pretrained(
        cls,
        checkpoint_path: str,
        step: Optional[int] = None,
    ):
        """Loads a model from a checkpoint that was saved via `save_pretrained`.

        Args:
            checkpoint_path (str): A path to either a directory of checkpoints or a single checkpoint.
            step (int, optional): If multiple checkpoints are present, which one to load. Defaults to the latest.
        """
        # load config
        with tf.io.gfile.GFile(
            tf.io.gfile.join(checkpoint_path, "config.json"), "r"
        ) as f:
            config = json.load(f)

        # load example batch
        with tf.io.gfile.GFile(
            tf.io.gfile.join(checkpoint_path, "example_batch.msgpack"), "rb"
        ) as f:
            example_batch = flax.serialization.msgpack_restore(f.read())

        logging.debug(
            "Model was trained with observations: %s",
            flax.core.pretty_repr(
                jax.tree_map(jnp.shape, example_batch["observation"])
            ),
        )
        logging.debug(
            "Model was trained with tasks: %s",
            flax.core.pretty_repr(jax.tree_map(jnp.shape, example_batch["task"])),
        )

        # load dataset statistics
        with tf.io.gfile.GFile(
            tf.io.gfile.join(checkpoint_path, "dataset_statistics.json"), "r"
        ) as f:
            dataset_statistics = json.load(f)
            dataset_statistics = jax.tree_map(
                np.array, dataset_statistics, is_leaf=lambda x: not isinstance(x, dict)
            )

        rng = jax.random.PRNGKey(0)
        # manually add token embeddings
        if "token_embedding" not in example_batch["task"]["language_instruction"]:
            example_batch["task"]["language_instruction"]["token_embedding"] = np.zeros((*example_batch["task"]["language_instruction"]["input_ids"].shape, 768))

        base_net = BaseNetwork(**config["base_net_kwargs"], octo_kwargs=config["model"])

        params_shape = jax.eval_shape(
            base_net.init, jax.random.PRNGKey(0), example_batch
        )["params"]
        # restore params, checking to make sure the shape matches
        checkpointer = orbax.checkpoint.CheckpointManager(
            checkpoint_path, orbax.checkpoint.PyTreeCheckpointer()
        )
        step = step if step is not None else checkpointer.latest_step()
        params = checkpointer.restore(step, params_shape)

        if config["text_processor"] is not None:
            text_processor = ModuleSpec.instantiate(config["text_processor"])()
        else:
            text_processor = None

        return cls(
            base_net=base_net,
            params=params,
            text_processor=text_processor,
            example_batch=example_batch,
            config=config,
            dataset_statistics=dataset_statistics,
        )

    def save_pretrained(
        self,
        step: int,
        checkpoint_path: Optional[str] = None,
        checkpoint_manager: Optional[orbax.checkpoint.CheckpointManager] = None,
    ):
        """Saves a model, as well as corresponding metadata needed for `load_pretrained`. Takes either a
        pre-existing checkpoint manager (which already knows where to save the checkpoint) or a path to a
        directory to save the checkpoint to.

        Args:
            step (int): Step number.
            checkpoint_path (str, optional): Path to save the checkpoint.
            checkpoint_manager (optional): Checkpoint manager to save the checkpoint.
            params (optional): Params to save. If None, uses self.params.
        """
        if (checkpoint_path is None) == (checkpoint_manager is None):
            raise ValueError(
                "Must provide exactly one of checkpoint_path or checkpoint_manager."
            )
        if checkpoint_manager is None:
            checkpoint_manager = orbax.checkpoint.CheckpointManager(
                checkpoint_path, orbax.checkpoint.PyTreeCheckpointer()
            )
        if checkpoint_path is None:
            checkpoint_path = str(checkpoint_manager._directory)

        # save params
        checkpoint_manager.save(
            step,
            self.params,
            {"save_args": orbax_utils.save_args_from_target(self.params)},
        )

        if jax.process_index() == 0:
            # save config
            config_path = tf.io.gfile.join(checkpoint_path, "config.json")
            if not tf.io.gfile.exists(config_path):
                with tf.io.gfile.GFile(config_path, "w") as f:
                    json.dump(self.config, f)

            # save example batch
            example_batch_path = tf.io.gfile.join(
                checkpoint_path, "example_batch.msgpack"
            )
            if not tf.io.gfile.exists(example_batch_path):
                with tf.io.gfile.GFile(example_batch_path, "wb") as f:
                    f.write(flax.serialization.msgpack_serialize(self.example_batch))

            # save dataset statistics
            dataset_statistics_path = tf.io.gfile.join(
                checkpoint_path, "dataset_statistics.json"
            )
            if not tf.io.gfile.exists(dataset_statistics_path):
                with tf.io.gfile.GFile(dataset_statistics_path, "w") as f:
                    json.dump(
                        jax.tree_map(lambda x: x.tolist(), self.dataset_statistics),
                        f,
                    )

    @classmethod
    def from_config(
        cls,
        config: Config,
        example_batch: Data,
        text_processor: Optional[Any] = None,
        rng: Optional[PRNGKey] = None,
        dataset_statistics: Optional[Data] = None,
    ):
        """Initializes a model with a fresh set of weights from a given config + example_batch.

        Args:
            config (Dict[str, Any]): Config dict. The only required key is "model", but other configuration
                may be saved for posterity.
            example_batch (Dict[str, Any]): Example batch.
            text_processor (Any, optional): Preprocessor for text inputs.
            rng (Optional[PRNGKey], optional): RNG key for initializing the model.
            dataset_statistics (Optional[Dict[str, Any]], optional): Dataset statistics.
        """
        rng = rng if rng is not None else jax.random.PRNGKey(0)
        example_batch = multihost_utils.process_allgather(example_batch)
        example_batch = jax.tree_map(lambda x: x[:1], example_batch)

        base_net = BaseNetwork(**config["base_net_kwargs"], octo_kwargs=config["model"])
        params = base_net.init(rng, example_batch)["params"]

        return cls(
            base_net=base_net,
            params=params,
            text_processor=text_processor,
            example_batch=example_batch,
            config=config,
            dataset_statistics=dataset_statistics,
        )
