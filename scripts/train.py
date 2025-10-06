# WARNING: importing tensorflow too late can silence important logging (╯°□°)╯︵ ┻━┻
import os
os.environ["TF_CUDNN_USE_AUTOTUNE"] = "1"
import tensorflow as tf

# isort: split

import datetime
from functools import partial
# import os
import os.path as osp
import pickle
import numpy as np
import random

from absl import app, flags, logging
from flax.traverse_util import flatten_dict
import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from ml_collections import config_flags
import optax
import tqdm
import wandb
import flax.jax_utils as flax_utils

import octo
from octo.data.dataset import make_interleaved_dataset, make_single_dataset
from octo.data.oxe import make_oxe_dataset_kwargs_and_weights
from octo.model.octo_model import OctoModel
from octo.utils import jax_utils
from octo.utils.spec import ModuleSpec
from octo.utils.train_callbacks import (
    RolloutVisualizationCallback,
    SaveCallback,
    ValidationCallback,
    VisualizationCallback,
)
from octo.utils.train_utils import (
    create_optimizer,
    filter_eval_datasets,
    format_name_with_config,
    process_text,
    Timer,
    TrainState,
)
from octo.utils.typing import Data

from hypervla.model import HyperVLA
from hypervla.utils import *
from hypervla.components.hypernetwork import HyperNetwork
from hypervla.components.base_network import BaseNetwork

# TF_CUDNN_USE_AUTOTUNE=1
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

FLAGS = flags.FLAGS

flags.DEFINE_string("name", "experiment", "Experiment name.")
flags.DEFINE_bool("debug", False, "Debug config (no wandb logging)")
flags.DEFINE_bool("offline", False, "Wandb log online or offline")
flags.DEFINE_string("wandb_tag", "", "Tags for wandb.")

config_dir = os.path.join(os.path.dirname(__file__), "configs")
config_flags.DEFINE_config_file(
    "config",
    os.path.join(config_dir, "config.py:transformer_bc"),
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

num_devices = jax.device_count()


def main(_):
    tf.random.set_seed(FLAGS.config.seed)
    np.random.seed(FLAGS.config.seed)
    random.seed(FLAGS.config.seed)
    tf.keras.utils.set_random_seed(FLAGS.config.seed)

    jax_utils.initialize_compilation_cache()

    assert FLAGS.config.dataset_kwargs.batch_size % jax.device_count() == 0
    # assert FLAGS.config.viz_kwargs.eval_batch_size % jax.device_count() == 0
    # assert FLAGS.config.dataset_kwargs.batch_size % jax.process_count() == 0
    # assert FLAGS.config.viz_kwargs.eval_batch_size % jax.process_count() == 0

    # create a 1D mesh with a single axis named "batch"
    mesh = Mesh(jax.devices(), axis_names="batch")
    # replicated sharding -- does not shard arrays
    replicated_sharding = NamedSharding(mesh, PartitionSpec())
    # data-parallel sharding -- shards arrays along the first axis
    dp_sharding = NamedSharding(mesh, PartitionSpec("batch"))

    def shard(batch):
        return multihost_utils.host_local_array_to_global_array(
            batch, mesh, PartitionSpec("batch")
        )

    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    if FLAGS.config.get("wandb_resume_id", None) is None:
        name = FLAGS.name
        wandb_id = "{time}_{name}".format(
            name=name,
            time=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
        wandb_id = jax_utils.host_broadcast_str(wandb_id)
        if jax.process_index() == 0:
            if FLAGS.debug:
                wandb_mode = "disabled"
            else:
                if FLAGS.offline:
                    wandb_mode = "offline"
                else:
                    wandb_mode = None
            tags = FLAGS.wandb_tag.split("+")
            wandb.init(
                config=FLAGS.config.to_dict(),
                id=wandb_id,
                name=name,
                mode=wandb_mode,
                tags=tags,
                **FLAGS.config.wandb,
            )

        if FLAGS.config.save_dir is not None:
            save_dir = tf.io.gfile.join(
                FLAGS.config.save_dir,
                FLAGS.config.wandb.project,
                FLAGS.config.wandb.group or "",
                wandb_id,
            )
            logging.info("Saving to %s", save_dir)
            if jax.process_index() == 0:
                wandb.config.update(dict(save_dir=save_dir), allow_val_change=True)
        else:
            save_dir = None
            logging.info("save_dir not passed in, not saving checkpoints")
    else:
        # resume previous run
        wandb_run = wandb.Api().run(FLAGS.config.wandb_resume_id)
        if jax.process_index() == 0:
            wandb.init(
                project=wandb_run.project,
                id=wandb_run.id,
                entity=wandb_run.entity,
                resume="must",
            )
        save_dir = wandb_run.config["save_dir"]
        logging.info("Resuming run %s", FLAGS.config.wandb_resume_id)
    save_callback = SaveCallback(save_dir)

    if jax.process_index() == 0:
        codebase_directory = osp.abspath(osp.join(osp.dirname(octo.__file__), ".."))
        wandb.run.log_code(codebase_directory)

    # set up text tokenization (this needs to happen after batching but before sharding)
    if FLAGS.config.text_processor is None:
        text_processor = None
    else:
        text_processor = ModuleSpec.instantiate(FLAGS.config.text_processor)()

    from octo.utils.train_utils import hf_weights_loader
    from octo.model.components.tokenizers import LanguageTokenizer
    # load token encoder (discrete tokens -> embedding vectors)
    language_token_encoder = LanguageTokenizer('t5-base', finetune_encoder=False)
    rng = jax.random.PRNGKey(FLAGS.config.seed)
    dummy_task = {"language_instruction": {"input_ids": np.ones((1, 16), dtype=np.int32), "attention_mask": np.ones((1, 16))}}
    t5_params = language_token_encoder.init(rng, dict(), dummy_task, train=False)['params']
    # Load pretrained weights
    t5_params = hf_weights_loader(t5_params, hf_model="t5-base")

    def process_batch(batch):
        batch = process_text(batch, text_processor)
        del batch["dataset_name"]
        return batch

    if FLAGS.config["base_net_kwargs"]["vit_kwargs"]["fine_tune_pretrained_image_encoder"] or FLAGS.config["hypernet_kwargs"].get("use_initial_image", False):
        from transformers import FlaxDinov2Model
        pretrained_image_encoder = FlaxDinov2Model.from_pretrained("facebook/dinov2-base")
        pretrained_params = pretrained_image_encoder.params

    def DINO_encode_image(raw_images):
        raw_images = raw_images / 255.0
        DINO_image_mean = jnp.array([0.485, 0.456, 0.406])
        DINO_image_std = jnp.array([0.229, 0.224, 0.225])
        raw_images = (raw_images - DINO_image_mean[None, None, None]) / DINO_image_std[None, None, None]
        raw_images = raw_images.transpose(0, 3, 1, 2)
        DINO_outputs = pretrained_image_encoder(pixel_values=raw_images, output_attentions=True)
        return DINO_outputs

    # load datasets
    if "oxe_kwargs" in FLAGS.config.dataset_kwargs:
        # create dataset_kwargs_list from oxe_kwargs
        (
            FLAGS.config.dataset_kwargs["dataset_kwargs_list"],
            FLAGS.config.dataset_kwargs["sample_weights"],
        ) = make_oxe_dataset_kwargs_and_weights(
            **FLAGS.config.dataset_kwargs["oxe_kwargs"],
            skip_unlabeled=FLAGS.config.dataset_kwargs["traj_transform_kwargs"].get("skip_unlabeled", False),
        )
        del FLAGS.config.dataset_kwargs["oxe_kwargs"]

        FLAGS.config.dataset_kwargs.batch_size //= jax.process_count()
        train_data = make_interleaved_dataset(**FLAGS.config.dataset_kwargs, train=True)

        train_data_iter = map(
            process_batch,
            train_data.iterator(prefetch=FLAGS.config.prefetch_num_batches),
        )
        # setup validation kwargs
        val_datasets_kwargs_list, _ = filter_eval_datasets(
            FLAGS.config.dataset_kwargs["dataset_kwargs_list"],
            FLAGS.config.dataset_kwargs["sample_weights"],
            FLAGS.config.eval_datasets,
        )
    else:
        standardize_fn = ModuleSpec.create(FLAGS.config["dataset_kwargs"]["dataset_kwargs"]["standardize_fn"])
        del FLAGS.config["dataset_kwargs"]["dataset_kwargs"]["standardize_fn"]
        FLAGS.config["dataset_kwargs"]["dataset_kwargs"]["standardize_fn"] = standardize_fn

        train_data = make_single_dataset(
            FLAGS.config.dataset_kwargs.dataset_kwargs,
            traj_transform_kwargs=FLAGS.config.dataset_kwargs.traj_transform_kwargs,
            frame_transform_kwargs=FLAGS.config.dataset_kwargs.frame_transform_kwargs,
            train=True,
        )
        train_data_iter = (
            train_data.repeat()
            .unbatch()
            .shuffle(FLAGS.config.dataset_kwargs.shuffle_buffer_size, seed=FLAGS.config.seed)
            .batch(FLAGS.config.dataset_kwargs.batch_size)
            .iterator()
        )
        train_data_iter = map(process_batch, train_data_iter)
        # setup validation kwargs
        val_datasets_kwargs_list = [FLAGS.config.dataset_kwargs.dataset_kwargs]

    example_batch = next(train_data_iter)
    example_batch['task']['language_instruction']['token_embedding'] = language_token_encoder.apply({'params': t5_params}, dict(), example_batch["task"], train=False).tokens
    example_batch["task"].pop("instruction_string")
    if FLAGS.config["hypernet_kwargs"].get("use_initial_image", False):
        DINO_outputs = DINO_encode_image(example_batch["initial_state"]["image_primary"].squeeze(1))
        example_batch["initial_state"]["patch_embeddings"] = jax.lax.stop_gradient(DINO_outputs.last_hidden_state)

    if FLAGS.config["base_net_kwargs"]["vit_kwargs"]["encoder_type"] == "Siglip":
        import torch
        from transformers import AutoImageProcessor, SiglipVisionModel
        siglip_model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224").cuda(0)
        processor = AutoImageProcessor.from_pretrained("google/siglip-base-patch16-224")

        inputs = processor(images=example_batch["observation"]["image_primary"].squeeze(1), return_tensors="pt")
        inputs = {k: v.cuda(0) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = siglip_model(**inputs)
        example_batch["observation"]["patch_embeddings"] = outputs.last_hidden_state.cpu().numpy()

    logging.info(f"Batch size: {example_batch['action'].shape[0]}")
    logging.info(f"Number of devices: {jax.device_count()}")
    logging.info(
        f"Batch size per device: {example_batch['action'].shape[0] // jax.device_count()}"
    )

    # setup random seed
    rng = jax.random.PRNGKey(FLAGS.config.seed)
    rng, init_rng = jax.random.split(rng)

    # set up HyperVLA and initialize weights
    model = HyperVLA.from_config(
        FLAGS.config.to_dict(),
        example_batch,
        rng=init_rng,
        dataset_statistics=train_data.dataset_statistics,
    )

    if FLAGS.config.pretrained_checkpoint_path is not None:
        with open(f"{FLAGS.config.pretrained_checkpoint_path}/{FLAGS.config.pretrained_checkpoint_step}/EMA_params.pkl", "rb") as f:
            EMA_params = pickle.load(f)
        model = model.replace(params=EMA_params[f"EMA_0.999"])
        del EMA_params

    # hardcode whether each param block in the HN is used for generation or shared in the base network
    def check_param_type(path, _):
        if "image_encoder" in path[0].key:
            return "shared"
        return "generated"

    HN_param_type = jax.tree_util.tree_map_with_path(check_param_type, model.params)

    # create optimizer
    tx, lr_callable, base_lr_callable, param_norm_callable = create_optimizer(
        model.params,
        HN_param_type,
        **FLAGS.config.optimizer.to_dict(),
    )

    # create train state
    train_state = TrainState.create(rng, model, tx)

    if FLAGS.config.get("wandb_resume_id", None) is not None:
        train_state = save_callback.state_checkpointer.restore(
            save_callback.state_checkpointer.latest_step(), items=train_state
        )
        checkpoint_step = int(train_state.step)
        logging.info("Restored checkpoint from %s", save_dir)
        if FLAGS.config.start_step is not None:
            start_step = FLAGS.config.start_step  # start_step overrides checkpoint
        else:
            start_step = checkpoint_step
        logging.info("Starting training from step %d", start_step)
    else:
        start_step = FLAGS.config.start_step or 0
    train_state = train_state.replace(step=start_step)

    # refreshes the train state so it doesn't crash w/ certain pre-trained loaders
    train_state = jax.device_get(train_state)
    
    if num_devices > 1:
        train_state = flax_utils.replicate(train_state)

    def sample_loss_fn(params, sample_data, dropout_rng, step):
        # add a batch dimension for each sample
        sample_data = jax.tree_map(lambda x: jnp.expand_dims(x, 0), sample_data)
        if FLAGS.config["hypernet_kwargs"].get("use_initial_image", False):
            initial_states = sample_data["initial_state"]
        else:
            initial_states = None
        dict_base_params, context_embedding = model.hypernet.apply({'params': params}, sample_data['task'], train=True, initial_states=initial_states, rngs={'dropout': dropout_rng})
        # squeeze the batch dimension, which is 1
        dict_base_params = jax.tree_map(lambda p: p.squeeze(0), dict_base_params)
        if FLAGS.config["base_net_kwargs"]["vit_kwargs"]["encoder_type"] == "EfficientNet":
            dropout_rng, drop_connect_rng = jax.random.split(dropout_rng)
            bound_base_net = model.base_net.bind({"params": dict_base_params}, rngs={"dropout": dropout_rng, "drop_connect": drop_connect_rng})
        else:
            if FLAGS.config["base_net_kwargs"]["vit_kwargs"].get("image_embedding_noise", 0.) > 0.:
                dropout_rng, embedding_noise_rng = jax.random.split(dropout_rng)
                rngs = {"dropout": dropout_rng, "embedding_noise": embedding_noise_rng}
            else:
                rngs = {"dropout": dropout_rng}
            bound_base_net = model.base_net.bind({"params": dict_base_params}, rngs=rngs)
        loss, metrics, policy_attention_map = bound_base_net.loss(sample_data, train=True)

        # reduce attention entropy of the action token
        # TODO: warmup and then anneal the coefficient for entropy loss?
        if FLAGS.config.auxiliary_loss.attention_entropy > 0.0:
            attention_prob = policy_attention_map[:, :, -1]
            epsilon = 1e-8
            log_prob = jnp.log(attention_prob + epsilon)
            # log_prob_clipped = jnp.maximum(attention_prob, 1e-8)  # for numerical stability
            per_head_entropy = -jnp.sum(attention_prob * log_prob, axis=-1)
            entropy_loss = jnp.mean(per_head_entropy)
            loss = loss + FLAGS.config.auxiliary_loss.attention_entropy * entropy_loss
            metrics["attention_entropy_loss"] = jax.lax.stop_gradient(entropy_loss)

        # alignment loss on attention map
        if FLAGS.config.auxiliary_loss.attention_map_alignment > 0.0:
            # attention map shape: batch_size * head_num * seq_len * seq_len
            # for policy, the last token is the action token
            policy_attention_map = policy_attention_map[:, :, -1, :-1]
            # for DINO, the first token is the class token
            reference_attention_map = jax.lax.stop_gradient(sample_data["observation"]["DINO_last_layer_attention_map"][:, :, 0, 1:])
            # alignment loss
            alignment_loss = ((policy_attention_map.mean(1) - reference_attention_map.mean(1)) ** 2).mean()
            # TODO: annealing startegy for alignment loss weight
            annealing_factor = 1. - step / FLAGS.config.num_steps
            alignment_loss_weight = annealing_factor * FLAGS.config.auxiliary_loss.attention_map_alignment
            loss = loss + alignment_loss_weight * alignment_loss
            metrics["attention_alignment_loss"] = jax.lax.stop_gradient(alignment_loss)
        # base reg
        # if FLAGS.config.optimizer.weight_decay_strategy != 'v4' and FLAGS.config.optimizer.get("base_weight_decay", 0.) > 0.:
        #     mask = jax.tree_util.tree_map_with_path(lambda path, _: "kernel" in jax.tree_util.keystr(path), dict_base_params)
        #     masked_base_params = jax.tree_map(lambda p, m: p * m, dict_base_params, mask)
        #     reg_term = optax.tree_utils.tree_l2_norm(masked_base_params, squared=True)
        #     reg_term *= FLAGS.config.optimizer.base_weight_decay
        #     # base_norm_square = jax.tree_map(lambda p, m: (p ** 2).sum() * m, dict_base_params, mask)
        #     # reg_term = FLAGS.config.auxiliary_loss.base_weight_decay * sum(jax.tree_util.tree_leaves(base_norm_square))
        #     loss += reg_term
        # metrics["base_weight_decay"] = reg_term
        metrics["base_params_norm"] = optax.global_norm(dict_base_params)
        # metrics["DINO_param_norm"] = optax.global_norm(dict_base_params["encoder"]["image_encoder"])
        # metrics["policy_param_norm"] = jnp.sqrt(metrics["base_params_norm"] ** 2 - metrics["DINO_param_norm"] ** 2)
        return loss, metrics

    def sample_weight_decay_loss(params, sample_data, dropout_rng):
        # add a batch dimension for each sample
        sample_data = jax.tree_map(lambda x: jnp.expand_dims(x, 0), sample_data)
        if FLAGS.config["hypernet_kwargs"].get("use_initial_image", False):
            initial_states = sample_data["initial_state"]
        else:
            initial_states = None
        dict_base_params = model.hypernet.apply({'params': params}, sample_data['task'], train=True, initial_states=initial_states, rngs={'dropout': dropout_rng})
        # squeeze the batch dimension, which is 1
        dict_base_params = jax.tree_map(lambda p: p.squeeze(0), dict_base_params)
        # base reg
        mask = jax.tree_util.tree_map_with_path(lambda path, _: "kernel" in jax.tree_util.keystr(path), dict_base_params)
        base_norm_square = jax.tree_map(lambda p, m: (p ** 2).sum() * m, dict_base_params, mask)
        loss = 0.5 * sum(jax.tree_util.tree_leaves(base_norm_square))
        return loss

    @partial(jax.pmap, axis_name='batch', backend='gpu', donate_argnums=(0, ))
    def train_step_pmap(state: TrainState, batch: Data, task_index):
        rephrase_strategy = FLAGS.config["auxiliary_loss"].get("rephrase_strategy", None)
        if rephrase_strategy == "replace":
            batch['rephrased_task']['language_instruction']['token_embedding'] = language_token_encoder.apply({'params': t5_params}, dict(), batch["rephrased_task"], train=True).tokens
            batch["task"]["language_instruction"] = batch['rephrased_task']['language_instruction']
        elif rephrase_strategy == "auxiliary_loss":
            batch['rephrased_task']['language_instruction']['token_embedding'] = language_token_encoder.apply({'params': t5_params}, dict(), batch["rephrased_task"], train=True).tokens
            batch['task']['language_instruction']['token_embedding'] = language_token_encoder.apply({'params': t5_params}, dict(), batch["task"], train=True).tokens
        else:
            batch['task']['language_instruction']['token_embedding'] = language_token_encoder.apply({'params': t5_params}, dict(), batch["task"], train=True).tokens
        # encode initial image if needed
        if FLAGS.config["hypernet_kwargs"].get("use_initial_image", False):
            DINO_outputs = DINO_encode_image(batch["initial_state"]["image_primary"].squeeze(1))
            batch["initial_state"]["patch_embeddings"] = jax.lax.stop_gradient(DINO_outputs.last_hidden_state)
        # compute DINOv2 image embedding if needed
        if FLAGS.config["base_net_kwargs"]["vit_kwargs"]["encoder_type"] == "DINOv2" and FLAGS.config.auxiliary_loss.attention_map_alignment > 0.0:
            # Merge the batch and window size dimension, as DINOv2 only accepts 4-dim inputs
            # raw_images = batch["observation"]["image_primary"].transpose(0, 1, 4, 2, 3)
            # raw_images = raw_images.reshape(batch_size * raw_images.shape[1], *raw_images.shape[2:])
            # TODO: squeeze window size dimension for now
            raw_images = batch["observation"]["image_primary"].squeeze(1)
            raw_images = raw_images / 255.0
            DINO_image_mean = jnp.array([0.485, 0.456, 0.406])
            DINO_image_std = jnp.array([0.229, 0.224, 0.225])
            raw_images = (raw_images - DINO_image_mean[None, None, None]) / DINO_image_std[None, None, None]
            raw_images = raw_images.transpose(0, 3, 1, 2)
            DINO_outputs = pretrained_image_encoder(pixel_values=raw_images, output_attentions=True)
            # drop the class token embedding
            # image_embeddings = image_embeddings[:, 1:]
            # move window_size to the sequence length dimension
            # image_embeddings = image_embeddings.reshape(batch_size, -1, image_embeddings.shape[-1])
            batch["observation"]["image_embedding"] = jax.lax.stop_gradient(DINO_outputs.last_hidden_state[:, 1:])
            batch["observation"]["DINO_last_layer_attention_map"] = DINO_outputs.attentions[-1]

        rng, dropout_rng = jax.random.split(state.rng)
        per_device_bs = batch['action'].shape[0]
        dropout_rngs = jax.random.split(dropout_rng, per_device_bs)

        # (losses, metrics), grads = jax.vmap(
        #     jax.value_and_grad(sample_loss_fn, has_aux=True),
        #     in_axes=(None, 0, 0)
        # )(state.model.params, batch, dropout_rngs)

        # grads = jax.tree_map(lambda g: g.mean(axis=0), grads)
        # grads = jax.lax.pmean(grads, axis_name='batch')
        # grad_norm = optax.global_norm(grads)

        def _loss_fn(params):
            losses, metrics = jax.vmap(sample_loss_fn, in_axes=(None, 0, 0, None))(params, batch, dropout_rngs, state.step)
            for task_name in task_index:
                metrics[f"task_loss_{task_name}"] = (losses * task_index[task_name]).sum()
            return losses.mean(), metrics

        (losses, metrics), grads = jax.value_and_grad(_loss_fn, has_aux=True)(state.model.params)
        grads = jax.lax.pmean(grads, axis_name='batch')

        updates, new_opt_state = state.tx.update(grads, state.opt_state, state.model.params)

        # For AdamW update on DINO params, need to minimize the distance to the original params instead of 0
        def delta_change_decay(path, params):
            HN_block_name = "encoder_image_encoder_" + "_".join([p.key for p in path])
            updates[HN_block_name] = updates[HN_block_name] + coefficient * params.ravel()

        if FLAGS.config["base_net_kwargs"]["vit_kwargs"]["fine_tune_pretrained_image_encoder"] and FLAGS.config["optimizer"]["base_weight_decay"] > 0:
            coefficient = base_lr_callable(state.step) * FLAGS.config.optimizer.base_weight_decay
            jax.tree_util.tree_map_with_path(delta_change_decay, pretrained_params)

        if FLAGS.config.optimizer.weight_decay_strategy == 'v4':
            weight_decay_losses, weight_decay_grads = jax.vmap(
                jax.value_and_grad(sample_weight_decay_loss, has_aux=False),
                in_axes=(None, 0, 0)
            )(state.model.params, batch, dropout_rngs)
            weight_decay_grads = jax.tree_map(lambda g: g.mean(axis=0), weight_decay_grads)
            weight_decay_grads = jax.lax.pmean(weight_decay_grads, axis_name='batch')
            # clip gradient
            weight_decay_grad_norm = optax.global_norm(weight_decay_grads)
            weight_decay_updates = jax.tree_map(lambda x: x / weight_decay_grad_norm * jnp.minimum(weight_decay_grad_norm, FLAGS.config.optimizer.clip_gradient), weight_decay_grads)
            # multiple by weight decay coefficient and learning rate
            coefficient = lr_callable(state.step) * FLAGS.config.auxiliary_loss.base_weight_decay
            weight_decay_updates = jax.tree_map(lambda x: coefficient * x, weight_decay_updates)
            updates = jax.tree_map(lambda x, y: x - y, updates, weight_decay_updates)

        # clip update norm
        # if FLAGS.config["optimizer"].get("clip_gradient", None) is not None:
        #     max_norm = FLAGS.config["optimizer"]["clip_gradient"]
        #     update_norm = optax.global_norm(updates)
        #     updates = jax.tree_map(lambda x: x / jnp.maximum(update_norm, max_norm) * max_norm, updates)

        grad_norm = optax.global_norm(grads)
        update_norm = optax.global_norm(updates)

        # losses_mean = losses.mean()
        losses_mean = jax.lax.pmean(losses, axis_name='batch')

        # ……………………………………………………………… #
        # caution, state.step might be a scalar int, rather than array,
        # So we directly call lr_callable(state.step) do not use indexing operation [0]
        # ……………………………………………………………… #

        # single_replica_params = jax.tree_map(lambda x: x[0], state.model.params)  # take the 0th replicate
        info = {
            "training_loss": losses_mean,
            "grad_norm": grad_norm,
            "update_norm": update_norm,
            "param_norm": param_norm_callable(state.model.params),
            "learning_rate": lr_callable(state.step),
        }

        # def compute_DINO_param_delta(path, params):
        #     HN_block_name = "encoder_image_encoder_" + "_".join([p.key for p in path])
        #     delta = state.model.params[HN_block_name] - params.ravel()
        #     return delta

        # DINO_param_delta = jax.tree_util.tree_map_with_path(compute_DINO_param_delta, pretrained_params)
        # info["DINO_param_delta_norm"] = optax.global_norm(DINO_param_delta)

        for task_name in task_index:
            task_loss = metrics.pop(f"task_loss_{task_name}")
            task_loss = jax.lax.psum(task_loss, axis_name='batch') / jax.lax.psum(task_index[task_name].sum(), axis_name='batch')
            info[f"task_loss_{task_name}"] = task_loss

        metrics_mean = jax.tree_map(lambda x: x.mean(), metrics)
        metrics_mean = jax.tree_map(lambda x: jax.lax.pmean(x, axis_name='batch'), metrics_mean)
        info.update(metrics_mean)

        if FLAGS.config.optimizer.weight_decay_strategy == 'v4':
            info["base_weight_decay_grad_norm"] = weight_decay_grad_norm

        new_params = optax.apply_updates(state.model.params, updates)
        new_model = state.model.replace(params=new_params)
        new_state = state.replace(
            model=new_model,
            opt_state=new_opt_state,
            rng=rng,
            step=state.step + 1  # step synchronous increase
        )
        return new_state, info

    # This function is only used for computing loss over a batch more efficiently for validation
    # The gradient can NOT work correctly for training
    def validation_action_loss(params, batch, rng, train=False):
        batch['task']['language_instruction']['token_embedding'] = language_token_encoder.apply({'params': t5_params}, dict(), batch["task"], train=train).tokens
        if FLAGS.config["hypernet_kwargs"].get("use_initial_image", False):
            DINO_outputs = DINO_encode_image(batch["initial_state"]["image_primary"].squeeze(1))
            batch["initial_state"]["patch_embeddings"] = jax.lax.stop_gradient(DINO_outputs.last_hidden_state)

        hypernet_rng, base_net_rng = jax.random.split(rng)
        if FLAGS.config["hypernet_kwargs"].get("use_initial_image", False):
            initial_states = batch["initial_state"]
        else:
            initial_states = None
        dict_base_params, _ = model.hypernet.apply({'params': params}, batch['task'], train=train, initial_states=initial_states, rngs={'dropout': hypernet_rng})

        def per_sample_predict_action(base_params, sample_data, dropout_rng):
            sample_data = jax.tree_map(lambda x: jnp.expand_dims(x, 0), sample_data)
            if FLAGS.config["base_net_kwargs"]["vit_kwargs"].get("image_embedding_noise", 0.) > 0.:
                dropout_rng, embedding_noise_rng = jax.random.split(dropout_rng)
                rngs = {"dropout": dropout_rng, "embedding_noise": embedding_noise_rng}
            else:
                rngs = {"dropout": dropout_rng}
            predicted_actions = model.base_net.apply(
                {"params": base_params},
                sample_data["observation"]["image_primary"],
                sample_data["task"]["language_instruction"]["token_embedding"],
                sample_data["observation"]["timestep_pad_mask"],
                train=train,
                rng=dropout_rng,
                rngs=rngs,
                method=BaseNetwork.predict_action, 
            )
            return predicted_actions

        dropout_rngs = jax.random.split(base_net_rng, batch["action"].shape[0])
        predicted_actions = jax.vmap(per_sample_predict_action, in_axes=(0, 0, 0))(dict_base_params, batch, dropout_rngs)
        target_actions = jnp.clip(batch["action"], -5., 5.)
        mse_loss = ((predicted_actions - target_actions) ** 2).mean() * FLAGS.config.base_net_kwargs.action_dim
        metrics = {"mse": mse_loss}
        return mse_loss, metrics

    val_callback = ValidationCallback(
        loss_fn=validation_action_loss,
        # process_batch_fn=lambda batch: shard(process_batch(batch)),
        process_batch_fn=lambda batch: process_batch(batch),
        text_processor=text_processor,
        val_dataset_kwargs_list=val_datasets_kwargs_list,
        dataset_kwargs=FLAGS.config.dataset_kwargs,
        **FLAGS.config.val_kwargs.to_dict(),
    )
    # viz_callback = VisualizationCallback(
    #     text_processor=text_processor,
    #     val_dataset_kwargs_list=val_datasets_kwargs_list,
    #     dataset_kwargs=FLAGS.config.dataset_kwargs,
    #     **FLAGS.config.viz_kwargs.to_dict(),
    # )
    # if "rollout_kwargs" in FLAGS.config:
    #     rollout_kwargs = FLAGS.config.rollout_kwargs.to_dict()
    #     dataset_name = rollout_kwargs.pop("dataset_name")
    #     rollout_callback = RolloutVisualizationCallback(
    #         text_processor=text_processor,
    #         action_proprio_metadata=train_data.dataset_statistics[dataset_name],
    #         **rollout_kwargs,
    #     )
    # else:
    #     rollout_callback = None

    def wandb_log(info, step):
        if jax.process_index() == 0:
            wandb.log(flatten_dict(info, sep="/"), step=step)

    timer = Timer()
    per_device_batch_size = FLAGS.config.dataset_kwargs.batch_size // num_devices

    @jax.jit
    def compute_params_EMA(old_EMA_params, params):
        new_EMA_params = {
            # "EMA_0.9": jax.tree_map(lambda x, y: 0.9 * x + 0.1 * y, old_EMA_params["EMA_0.9"], params),
            # "EMA_0.99": jax.tree_map(lambda x, y: 0.99 * x + 0.01 * y, old_EMA_params["EMA_0.99"], params),
            "EMA_0.999": jax.tree_map(lambda x, y: 0.999 * x + 0.001 * y, old_EMA_params["EMA_0.999"], params),
        }
        return new_EMA_params

    for i in tqdm.tqdm(
        range(start_step, int(FLAGS.config.num_steps)),
        total=int(FLAGS.config.num_steps),
        initial=start_step,
        dynamic_ncols=True,
    ):  
        timer.tick("total")
        
        with timer("dataset"):
            batch = next(train_data_iter)

            if FLAGS.config["base_net_kwargs"]["vit_kwargs"]["encoder_type"] == "Siglip":
                inputs = processor(images=batch["observation"]["image_primary"].squeeze(1), return_tensors="pt")
                inputs = {k: v.cuda(0) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = siglip_model(**inputs)
                batch["observation"]["patch_embeddings"] = outputs.last_hidden_state.cpu().numpy()

            batch = jax.tree_map(
                lambda x: x.reshape(num_devices, per_device_batch_size, *x.shape[1:]),
                batch
            )

        if FLAGS.config.dataset_kwargs.traj_transform_kwargs.get("skip_unlabeled", False):
            # if not batch['task']['pad_mask_dict']['language_instruction'].all():
            if (batch['task']['language_instruction']['attention_mask'].sum(-1) <= 1).any():
                print (f"step {i} does not use language instruction")
                breakpoint()
        
        instruction_strings = batch["task"].pop("instruction_string")
        task_index = dict()
        for task_name in [b"close top drawer", b"close middle drawer", b"close bottom drawer"]:
            task_index[task_name.decode("utf-8")] = (instruction_strings == task_name)

        if "rephrased_task" in batch:
            for key in batch["task"]:
                if key != "language_instruction":
                    batch["rephrased_task"][key] = batch["task"][key]

        with timer("train"):
            # jax.debug.print("+++ before train_step_pmap")
            # jax.debug.print("batch['action'].shape: {}", batch['action'].shape)
            # jax.debug.print("batch['observation']['image_primary'].shape: {}", batch['observation']['image_primary'].shape)

            train_state, update_info = train_step_pmap(train_state, batch, task_index)
            update_info = flax_utils.unreplicate(update_info)

        timer.tock("total")

        if (i + 1) % FLAGS.config.optimizer.grad_accumulation_steps != 0:
            continue

        current_update_step = (i + 1) // FLAGS.config.optimizer.grad_accumulation_steps

        if FLAGS.config.get("save_param_EMA", False):
            if current_update_step == FLAGS.config.EMA_start_step:
                model_params = flax_utils.unreplicate(train_state.model.params)
                EMA_params = {
                    # "EMA_0.9": jax.tree_map(lambda x: jnp.array(x, copy=True) if isinstance(x, jnp.ndarray) else x, model_params),
                    # "EMA_0.99": jax.tree_map(lambda x: jnp.array(x, copy=True) if isinstance(x, jnp.ndarray) else x, model_params),
                    "EMA_0.999": jax.tree_map(lambda x: jnp.array(x, copy=True) if isinstance(x, jnp.ndarray) else x, model_params),
                }
            elif current_update_step > FLAGS.config.EMA_start_step:
                EMA_params = compute_params_EMA(EMA_params, flax_utils.unreplicate(train_state.model.params))
        
        if current_update_step % FLAGS.config.save_interval == 0:
            if num_devices > 1:
                save_callback(flax_utils.unreplicate(train_state), current_update_step)
            else:
                save_callback(train_state, current_update_step)
            if FLAGS.config.get("save_param_EMA", False) and current_update_step >= FLAGS.config.EMA_start_step:
                with open(f"{save_dir}/{current_update_step}/EMA_params.pkl", "wb") as f:
                    pickle.dump(EMA_params, f)

        if current_update_step % FLAGS.config.eval_interval == 0:
            logging.info("Evaluating...")
            with timer("eval"):
                if num_devices > 1:
                    val_metrics = val_callback(flax_utils.unreplicate(train_state), current_update_step)
                else:
                    val_metrics = val_callback(train_state, current_update_step)
                wandb_log(val_metrics, step=current_update_step)

        # if (i + 1) % FLAGS.config.viz_interval == 0:
        #     logging.info("Visualizing...")
        #     with timer("visualize"):
        #         viz_metrics = viz_callback(train_state, i + 1)
        #         wandb_log(viz_metrics, step=i + 1)

        #     if rollout_callback is not None:
        #         with timer("rollout"):
        #             rollout_metrics = rollout_callback(train_state, i + 1)
        #             wandb_log(rollout_metrics, step=i + 1)

        if current_update_step % FLAGS.config.log_interval == 0:
            update_info = jax.device_get(update_info)
            log_dict = {
                "training": update_info,
                "timer": timer.get_average_times(),
                # "task_transition_num": dict(),
            }
            # for task in [b"close top drawer", b"close middle drawer", b"close bottom drawer"]:
            #     task_transition_num = (instruction_strings == task).sum()
            #     log_dict["task_transition_num"][task.decode("utf-8")] = task_transition_num
            wandb_log(
                log_dict,
                step=current_update_step,
            )



if __name__ == "__main__":
    app.run(main)
