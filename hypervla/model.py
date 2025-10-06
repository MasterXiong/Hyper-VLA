from functools import partial
import json
import logging
from typing import Optional

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

from octo.utils.typing import Config, Data, Params, PRNGKey

from hypervla.components.base_network import *
from hypervla.components.hypernetwork import HyperNetwork, InitOptions
from hypervla.utils import *


@struct.dataclass
class HyperVLA:

    hypernet: HyperNetwork = struct.field(pytree_node=False)
    base_net: nn.Module = struct.field(pytree_node=False)
    config: Config = struct.field(pytree_node=False)
    params: Params
    base_net_metadata: dict
    example_batch: Data
    dataset_statistics: Optional[Data]

    def create_tasks(
        self, 
        goals: Optional[Data] = None, 
        instruction_dict: dict = None,
        initial_state = None, 
    ):
        """Creates tasks dict from goals and texts.

        Args:
            goals: if not None, dict of arrays with shape (batch_size, *)
            texts: if not None, list of texts of length batch_size

        Omit images to run the language-conditioned model, and omit texts to run the
        goal-conditioned model.
        """
        # task dict for octo base model
        tasks = {"pad_mask_dict": {}}
        # zero padding mask for image goals
        batch_size = instruction_dict["language_instruction"]["input_ids"].shape[0]
        tasks.update(
            {
                k: np.zeros((batch_size, *v.shape[1:]), dtype=v.dtype)
                for k, v in self.example_batch["task"].items()
                if k not in ("pad_mask_dict", "language_instruction")
            }
        )
        tasks["pad_mask_dict"].update(
            {
                k: np.zeros(batch_size, dtype=bool)
                for k in tasks.keys()
                if k != "pad_mask_dict"
            }
        )
        # pad mask for language instructions
        tasks["pad_mask_dict"]["language_instruction"] = np.ones(batch_size, dtype=bool)
        tasks["language_instruction"] = instruction_dict["language_instruction"]

        # generate base params
        (dict_base_params, _), intermediate_states = self.hypernet.apply(
            {'params': self.params}, 
            tasks, 
            train=False, 
            initial_states=initial_state,
            mutable=True,
            capture_intermediates=True,
        )
        dict_base_params = jax.tree_map(lambda p: p.squeeze(0), dict_base_params)

        return dict_base_params, tasks, intermediate_states

    @jax.jit
    def sample_actions(
        self,
        images,
        instruction_dict,
        task,
        timestep_pad_mask,
        base_params,
        # unnormalization_statistics: Optional[Data] = None,
        # normalization_type: NormalizationType = NormalizationType.NORMAL,
        train: bool = False,
        # argmax: bool = False,
        # sample_shape: Tuple[int, ...] = (),
        rng: Optional[PRNGKey] = None,
        # temperature: float = 1.0,
        image_embeddings = None, 
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
        # squeeze the horizon dimension for now
        images = images.squeeze(1)

        if self.config["base_net_kwargs"]["vit_kwargs"].get("image_embedding_noise", 0.) > 0.:
            dropout_rng, embedding_noise_rng = jax.random.split(rng)
            rngs = {"dropout": dropout_rng, "embedding_noise": embedding_noise_rng}
        else:
            rngs = {'dropout': rng}

        action, intermediate_states = self.base_net.apply(
            {'params': base_params}, 
            images, 
            instruction_dict["language_instruction"]["token_embedding"], 
            timestep_pad_mask, 
            rng=rng, 
            rngs=rngs, 
            train=train, 
            image_embeddings=image_embeddings, 
            method=BaseNetwork.predict_action, 
            mutable=['intermediates'], 
        )
        return action, intermediate_states

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
        
        if "action_head_kwargs" not in config['base_net_kwargs']:
            config['base_net_kwargs']["action_head_kwargs"] = dict(
                token_per_horizon=False,
                squash_continuous_action=True,
                clip_target=False,
                max_action=5., # should set to 1 when using bound norm
            )

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

        # setup the base network
        base_net, _, _, base_net_metadata = cls.init_base_net(config, example_batch, rng)

        # setup the hypernet
        hypernet = HyperNetwork(base_net_metadata, config["hypernet_kwargs"])

        if config["hypernet_kwargs"].get("use_initial_image", False):
            initial_states = example_batch["initial_state"]
        else:
            initial_states = None
        params_shape = jax.eval_shape(
            partial(hypernet.init, train=False), jax.random.PRNGKey(0), example_batch['task'], initial_states=initial_states
        )["params"]
        # restore params, checking to make sure the shape matches
        checkpointer = orbax.checkpoint.CheckpointManager(
            checkpoint_path, orbax.checkpoint.PyTreeCheckpointer()
        )
        step = step if step is not None else checkpointer.latest_step()
        params = checkpointer.restore(step, params_shape)

        return cls(
            hypernet=hypernet,
            base_net=base_net,
            params=params,
            base_net_metadata=base_net_metadata,
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
        rng: Optional[PRNGKey] = None,
        dataset_statistics: Optional[Data] = None,
    ):
        """Initializes a model with a fresh set of weights from a given config + example_batch.

        Args:
            config (Dict[str, Any]): Config dict. The only required key is "model", but other configuration
                may be saved for posterity.
            example_batch (Dict[str, Any]): Example batch.
            rng (Optional[PRNGKey], optional): RNG key for initializing the model.
            dataset_statistics (Optional[Dict[str, Any]], optional): Dataset statistics.
        """
        rng = rng if rng is not None else jax.random.PRNGKey(0)
        hypernet_rng, base_net_rng = jax.random.split(rng)
        example_batch = multihost_utils.process_allgather(example_batch)
        example_batch = jax.tree_map(lambda x: x[:1], example_batch)

        # setup the base network
        base_net, init_base_params, flatten_init_base_params, base_net_metadata = cls.init_base_net(config, example_batch, base_net_rng)
        # setup the hypernet
        hypernet = HyperNetwork(
            base_net_metadata, 
            config["hypernet_kwargs"], 
        )

        @jax.jit
        def _init(rng):
            dropout_rng, rng = jax.random.split(rng)
            if config["hypernet_kwargs"].get("use_initial_image", False):
                initial_states = example_batch["initial_state"]
            else:
                initial_states = None
            return hypernet.init({'params': rng, 'dropout': dropout_rng}, example_batch['task'], train=True, initial_states=initial_states)

        params = _init(hypernet_rng)["params"]

        generation_strategy = config["hypernet_kwargs"].get("generation_strategy", "full")
        if generation_strategy == "block":
            # bias init
            def bias_init(path, value, generation_flag):
                path_str = '_'.join([x.key for x in path])
                # only apply to param blocks that require bias init
                if base_net_metadata["output_head_info"][path_str]["init_strategy"] == InitOptions.VARIANCE_INIT:
                    return
                if generation_flag:
                    if config["hypernet_kwargs"].get("share_TF_output_head", False):
                        if 'encoderblock_' in path_str:
                            if 'encoderblock_0' in path_str:
                                path_str = path_str.replace('encoderblock_0', 'encoderblock')
                            else:
                                return
                    params[f'output_head_{path_str}']['bias'] = value.ravel()
                else:
                    params[path_str] = value.ravel()

            jax.tree_util.tree_map_with_path(bias_init, init_base_params, base_net_metadata["generation_flag"])

        else:
            if config["hypernet_kwargs"].get("output_head_bias", True):
                params['output_head']['bias'] = flatten_init_base_params
            else:
                rngs = jax.random.split(rng, params['output_head']['kernel'].shape[0])
                init_params = []
                for rng in rngs:
                    _, p, _ = cls.init_base_net(config, example_batch, rng)
                    init_params.append(p)
                init_params = np.stack(init_params)
                params['output_head']['kernel'] = init_params

        return cls(
            hypernet=hypernet,
            base_net=base_net,
            params=params,
            base_net_metadata=base_net_metadata,
            example_batch=example_batch,
            config=config,
            dataset_statistics=dataset_statistics,
        )

    @classmethod
    def init_base_net(cls, config, example_batch, rng):
        # setup the base network
        base_net = BaseNetwork(**config["base_net_kwargs"], octo_kwargs=config["model"])
        if config["base_net_kwargs"]["vit_kwargs"]["encoder_type"] == "EfficientNet":
            rng, dropout_rng, drop_connect_rng = jax.random.split(rng, 3)
            init_base_params = base_net.init({'params': rng, 'dropout': dropout_rng, 'drop_connect': drop_connect_rng}, example_batch)["params"]
            print(base_net.tabulate({'params': rng, 'dropout': dropout_rng, 'drop_connect': drop_connect_rng}, example_batch, depth=3))
        else:
            rng, dropout_rng = jax.random.split(rng)
            if config["base_net_kwargs"]["vit_kwargs"].get("image_embedding_noise", 0.) > 0.:
                dropout_rng, embedding_noise_rng = jax.random.split(dropout_rng)
                rngs = {'params': rng, "dropout": dropout_rng, "embedding_noise": embedding_noise_rng}
            else:
                rngs = {'params': rng, 'dropout': dropout_rng}
            init_base_params = base_net.init(rngs, example_batch)["params"]
            print(base_net.tabulate(rngs, example_batch, depth=3))
        flatten_init_base_params, _ = jax.tree_util.tree_flatten(init_base_params)
        flatten_init_base_params = np.concatenate([p.ravel() for p in flatten_init_base_params])

        base_param_shapes = jax.tree_map(lambda x: np.array(x.shape), init_base_params)
        base_param_dim = jax.tree_map(lambda x: np.prod(x).item(), base_param_shapes)

        # assign a layer token index to each leaf node of the base param dict
        # so that each node knows which layer token to use to generate its parameters
        index = 0
        token_index_dict = jax.tree_map(lambda _: 0, base_param_shapes)
        # mask out layer tokens that are not generated by HN
        shared_modules = config["hypernet_kwargs"].get("shared_modules", tuple())
        layer_token_mask = []
        if config["hypernet_kwargs"].get("share_layer_index", False):
            layer_token_mask = [True]
            index = 1
        else:
            # image encoder
            if config["base_net_kwargs"]["vit_kwargs"]["encoder_type"] == "SmallStem":
                for module in base_param_shapes["encoder"]["SmallStem_0"]:
                    token_index_dict["encoder"]["SmallStem_0"][module] = jax.tree_map(lambda _: index, base_param_shapes["encoder"]["SmallStem_0"][module])
                    index += 1
                    if "SmallStem_0" in shared_modules:
                        layer_token_mask.append(False)
                    else:
                        layer_token_mask.append(True)
            elif config["base_net_kwargs"]["vit_kwargs"]["encoder_type"] == "EfficientNet":
                assert "EfficientNet" in shared_modules, "Only support shared EfficientNet"
                token_index_dict["encoder"]["EfficientNet_0"] = jax.tree_map(lambda _: index, base_param_shapes["encoder"]["EfficientNet_0"])
                index += 1
                layer_token_mask.append(False)
            elif config["base_net_kwargs"]["vit_kwargs"]["encoder_type"] in ["DINOv2", "CLIP"]:
                assert "image_encoder" in shared_modules, "Only support shared params for pretrained image encoder"
                token_index_dict["encoder"]["image_encoder"] = jax.tree_map(lambda _: index, base_param_shapes["encoder"]["image_encoder"])
                index += 1
                layer_token_mask.append(False)
            # Transformer layers
            for module in base_param_shapes["encoder"]["Transformer_0"]:
                token_index_dict["encoder"]["Transformer_0"][module] = jax.tree_map(lambda _: index, base_param_shapes["encoder"]["Transformer_0"][module])
                index += 1
                layer_token_mask.append(True)
            for module in base_param_shapes["encoder"]:
                if module in ["SmallStem_0", "Transformer_0", "EfficientNet_0", "image_encoder"]:
                    continue
                token_index_dict["encoder"][module] = jax.tree_map(lambda _: index, base_param_shapes["encoder"][module])
                index += 1
                layer_token_mask.append(True)
            token_index_dict["action_head"] = jax.tree_map(lambda _: index, base_param_shapes["action_head"])
            index += 1
            layer_token_mask.append(True)

        # determine if each param block in the base net is generated or shared
        def filter(path, value):
            path_keys = [p.key for p in path]
            shared_modules = config["hypernet_kwargs"].get("shared_modules", tuple())
            for module in shared_modules:
                for path_key in path_keys:
                    if module in path_key:
                        return False
            return True

        if config["hypernet_kwargs"].get("share_all_params", False):
            generation_flag = jax.tree_map(lambda _: False, base_param_shapes)
        else:
            generation_flag = jax.tree_util.tree_map_with_path(filter, base_param_shapes)
        pretty_print_meta_data(generation_flag, token_index_dict)

        # load pre-trained params
        if config["base_net_kwargs"]["vit_kwargs"]["encoder_type"] == "DINOv2":
            DINOv2_weights_loader(init_base_params)
        if config["base_net_kwargs"]["vit_kwargs"]["encoder_type"] == "CLIP":
            CLIP_weights_loader(init_base_params)

        base_net_metadata = {
            'token_index_dict': token_index_dict,
            'block_num': index,
            'param_shape': base_param_shapes,
            'total_param_num': flatten_init_base_params.shape[0],
            'param_dim': base_param_dim,
            'generation_flag': generation_flag,
            'layer_token_mask': np.array(layer_token_mask),
        }

        # 1. determine the structure of the output heads
        def set_output_head_info(path, base_shape, base_dim, generated):
            # return a dict of: 
            # head output dim
            # initialization strategy
            # initialization variance for variance init (0 for other cases)
            output_head_info = dict()
            output_head_info["output_dim"] = base_dim
            output_head_info["generation_flag"] = generated
            path_string = '.'.join([p.key for p in path])
            init_strategy = config["hypernet_kwargs"].get("init_strategy", InitOptions.BIAS_INIT)
            # always use bias init for normalization layers
            if "encoder_norm" in path_string or "LayerNorm" in path_string:
                init_strategy = InitOptions.BIAS_INIT
            if "GroupNorm" in path_string:
                init_strategy = InitOptions.BIAS_INIT
            # always use bias init for shared params
            if not generated:
                init_strategy = InitOptions.BIAS_INIT
            output_head_info["init_strategy"] = init_strategy
            # determine variance for variance init
            if init_strategy == InitOptions.VARIANCE_INIT and path[-1].key != "bias":
                if path[-1].key == "pos_embedding":
                    variance = 0.02 ** 2
                elif path[-2].key == "out":
                    variance = 1. / (base_shape[0] * base_shape[1])
                else:
                    variance = 1. / base_shape[0]
                if not config["hypernet_kwargs"].get("scale_context_embedding", False):
                    variance = variance / config["hypernet_kwargs"]["context_embedding_dim"]
            else:
                variance = 0.
            output_head_info["init_variance"] = variance
            return output_head_info

        output_head_info = jax.tree_util.tree_map_with_path(set_output_head_info, base_param_shapes, base_param_dim, generation_flag)
        # output head sharing for Transformer layers in the base network
        if config["hypernet_kwargs"].get("share_TF_output_head", False):
            output_head_info['encoder']['Transformer_0']['encoderblock'] = output_head_info['encoder']['Transformer_0'].pop('encoderblock_0')
            for layer in range(1, config["base_net_kwargs"]["vit_kwargs"]["num_layers"]):
                del output_head_info['encoder']['Transformer_0'][f'encoderblock_{layer}']
        # flatten dict
        output_head_info = flatten_dict(output_head_info)
        base_net_metadata["output_head_info"] = output_head_info

        return base_net, init_base_params, flatten_init_base_params, base_net_metadata


def pretty_print_meta_data(generation_flag, token_index_dict):

    def print_node(sub_generation_flag, sub_token_index_dict, depth):
        prefix = '-' * depth * 2
        for key in sub_generation_flag:
            if type(sub_generation_flag[key]) is dict:
                print (f'{prefix}{key}')
                print_node(sub_generation_flag[key], sub_token_index_dict[key], depth + 1)
            else:
                print (f'{prefix}{key}: HN generated: {sub_generation_flag[key]}, context token index: {sub_token_index_dict[key]}')

    print_node(generation_flag, token_index_dict, 0)


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict) and "output_dim" not in v.keys():  # Recursively call flatten_dict for nested dictionaries
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def DINOv2_weights_loader(params):
    """Loads weights from a HuggingFace model into params."""
    from transformers import FlaxDinov2Model

    model = FlaxDinov2Model.from_pretrained("facebook/dinov2-base")

    model_variables = model.params
    replaced = False

    def find_and_replace(params, key, replacement):
        nonlocal replaced
        for k in params.keys():
            if k == key:
                params[k] = replacement
                print(f"Replaced {key} in params")
                replaced = True
                return
            if isinstance(params[k], type(params)):
                find_and_replace(params[k], key, replacement)

    find_and_replace(params, "image_encoder", model_variables)
    assert replaced, "Failed to load weights"
    return params


def CLIP_weights_loader(params):
    """Loads weights from a HuggingFace model into params."""
    from transformers import FlaxCLIPVisionModel

    model = FlaxCLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")

    model_def, model_variables = model.module, model.params
    replaced = False

    def find_and_replace(params, key, replacement):
        nonlocal replaced
        for k in params.keys():
            if k == key:
                params[k] = replacement
                print(f"Replaced {key} in params")
                replaced = True
                return
            if isinstance(params[k], type(params)):
                find_and_replace(params[k], key, replacement)

    find_and_replace(params, "image_encoder", model_variables)
    assert replaced, "Failed to load weights"
    return params