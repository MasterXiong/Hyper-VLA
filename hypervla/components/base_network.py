from flax import linen as nn

from octo.utils.spec import ModuleSpec

from hypervla.components.action_heads import *
from hypervla.components.base_cnn import CNN
from hypervla.components.base_vit import ViT
from hypervla.components.base_octo import OctoTransformer


class BaseNetwork(nn.Module):
    model_type: str
    action_head_type: str
    octo_kwargs: dict
    cnn_kwargs: dict
    vit_kwargs: dict
    action_head_kwargs: dict
    action_horizon: int = 4
    action_dim: int = 7

    def setup(self):
        # determine action token number
        if self.action_head_type == 'discrete':
            if self.action_head_kwargs["discrete_token_type"] == "action_dim_and_action_horizon":
                action_token_num = self.action_horizon * self.action_dim
            elif self.action_head_kwargs["discrete_token_type"] == "action_horizon":
                action_token_num = self.action_horizon
        else:
            if self.action_head_kwargs["token_per_horizon"]:
                action_token_num = self.action_horizon
            else:
                action_token_num = 1
        # base encoder
        if self.model_type == 'cnn':
            self.encoder = CNN(**self.cnn_kwargs)
        elif self.model_type == 'vit':
            self.encoder = ViT(**self.vit_kwargs, action_token_num=action_token_num)
        elif self.model_type == 'octo':
            use_pretrained_image_tokenizer = self.octo_kwargs.get("use_pretrained_image_tokenizer", False)
            # do not include image tokenizer in Octo model if we use a pretrained image tokenizer
            if use_pretrained_image_tokenizer:
                observation_tokenizer_defs = dict()
            else:
                observation_tokenizer_defs = {
                    k: ModuleSpec.instantiate(spec)()
                    for k, spec in self.octo_kwargs["observation_tokenizers"].items()
                }
            self.encoder = OctoTransformer(
                observation_tokenizers=observation_tokenizer_defs,
                readouts={"action": action_token_num},
                transformer_kwargs=self.octo_kwargs["transformer_kwargs"],
                token_embedding_size=self.octo_kwargs["token_embedding_size"],
                max_horizon=self.octo_kwargs["max_horizon"],
                repeat_task_tokens=self.octo_kwargs["repeat_task_tokens"],
                use_correct_attention=True,
                use_pretrained_image_tokenizer=use_pretrained_image_tokenizer,
            )
        else:
            raise NotImplementedError
        # base action head
        if self.action_head_type == 'diffusion':
            self.action_head = DiffusionActionHead(
                readout_key="readout_action",
                use_map=False,
                action_horizon=self.action_horizon,
                action_dim=self.action_dim,
                n_diffusion_samples=1,
                dropout_rate=self.action_head_kwargs.get("diffusion_dropout_rate", 0.0),
                num_blocks=self.action_head_kwargs.get("num_blocks", 3),
                hidden_dim=self.action_head_kwargs.get("hidden_dim", 256),
            )
        elif self.action_head_type == 'continuous':
            self.action_head = ContinuousActionHead(
                readout_key="readout_action",
                use_map=False,
                action_horizon=self.action_horizon,
                action_dim=self.action_dim,
                **self.action_head_kwargs,
            )
        elif self.action_head_type == 'mix':
            self.action_head = MixActionHead(
                readout_key="readout_action",
                use_map=False,
                action_horizon=self.action_horizon,
                action_dim=self.action_dim,
                max_action=self.action_head_kwargs.get("max_action", 5.),
                token_per_horizon=self.action_head_kwargs["token_per_horizon"],
                squash_continuous_action=self.action_head_kwargs["squash_continuous_action"],
                tanh_scaling_factor=self.action_head_kwargs.get("tanh_scaling_factor", 5.),
                clip_target=self.action_head_kwargs["clip_target"],
                hidden_dims=self.action_head_kwargs.get("hidden_dims", tuple()),
            )
        elif self.action_head_type == 'discrete':
            self.action_head = DiscreteActionHead(
                readout_key="readout_action",
                use_map=False,
                action_horizon=self.action_horizon,
                action_dim=self.action_dim,
                token_per=self.action_head_kwargs["discrete_token_type"],
            )
        else:
            raise NotImplementedError

    def encode(self, images, instruction_embeddings, train=True, image_embeddings=None):
        action_embedding, attention_map = self.encoder(
            images, 
            instruction_embeddings, 
            train=train,
            image_embeddings=image_embeddings,
        )
        # align with Octo's action head interface
        # add one more dimension for window size
        embedding_dict = {"readout_action": TokenGroup(action_embedding[:, None, :, :], None)}
        return embedding_dict, attention_map

    def __call__(self, batch):
        if self.model_type == 'vit':
            # TODO: squeeze the window size dimension for now
            images = batch["observation"]["image_primary"].squeeze(1)
            # language instruction embedding
            instruction_embeddings = batch["task"]["language_instruction"]["token_embedding"]
            if self.vit_kwargs["encoder_type"] == "Siglip":
                image_embeddings = batch["observation"]["patch_embeddings"]
            else:
                image_embeddings = None
            embedding_dict, attention_map = self.encode(images, instruction_embeddings, train=True, image_embeddings=image_embeddings)
        elif self.model_type == 'octo':
            embedding_dict = self.encoder(
                batch["observation"],
                batch["task"],
                batch["observation"]["timestep_pad_mask"],
                train=True,
            )
        actions = self.action_head(embedding_dict)
        return actions
    
    def loss(self, batch, train=True):
        if self.model_type == 'vit':
            # TODO: squeeze the window size dimension for now
            images = batch["observation"]["image_primary"].squeeze(1)
            # language instruction embedding
            instruction_embeddings = batch["task"]["language_instruction"]["token_embedding"]
            if self.vit_kwargs["encoder_type"] == "Siglip":
                image_embeddings = batch["observation"]["patch_embeddings"]
            else:
                image_embeddings = None
            embedding_dict, attention_map = self.encode(
                images, 
                instruction_embeddings, 
                train=train, 
                image_embeddings=image_embeddings, 
            )
        elif self.model_type == 'octo':
            embedding_dict = self.encoder(
                batch["observation"],
                batch["task"],
                batch["observation"]["timestep_pad_mask"],
                train=train,
            )
        # compute action loss
        loss, metrics = self.action_head.loss(
            embedding_dict,
            batch["action"],
            batch["observation"]["timestep_pad_mask"],
            batch["action_pad_mask"],
            train=train,
        )
        return loss, metrics, attention_map

    def predict_action(self, observation, task, timestep_pad_mask, rng, train=False, image_embeddings=None):
        if observation.shape[1] == 1:
            observation = observation.squeeze(1)
        embedding_dict, attention_map = self.encode(observation, task, train=train, image_embeddings=image_embeddings)

        # action head
        actions = self.action_head.predict_action(
            embedding_dict,
            rng=rng,
            train=train,
            argmax=True,
            temperature=1.0,
        )
        return actions
