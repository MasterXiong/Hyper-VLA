import jax
import jax.numpy as jnp
from flax import linen as nn

from octo.model.components.vit_encoders import *
from hypervla.components.transformer import Transformer
from hypervla.components.efficient_net import EfficientNet, MODEL_CONFIGS
from transformers import FlaxDinov2Model, FlaxCLIPVisionModel


# --------------------------------------------------- #
# Diagram for ViT:                                    #
#                                                     #
#           ┌──────────────────┐                      #
#           │    Input Image   │                      #
#           │    (256x256x3)   │                      #
#           └─────────┬────────┘                      #
#                     │                               #
#             [Conv to patches]                       #
#                     │                               #
#                     v                               #
#        ┌───────────────────────────┐                #
#        │ Patches: 256 total        │                #
#        │ Shape: [B,256,hidden_dim] │                #
#        └────────────┬──────────────┘                #
#                     │                               #
#               [Add CLS Token]                       #
#                     │                               #
#                     v                               #
#    ┌─────────────────────────────────┐              #
#    │ Sequence length = 1 (CLS) + 256 │              #
#    │ Shape: [B,257,hidden_dim]       │              #
#    └────────────────┬────────────────┘              #
#                     │                               #
#    [Add Positional Embedding + Dropout]             #
#                     │                               #
#                     v                               #
#    ┌────────────────────────────────────────────┐   #
#    │   Transformer Encoder (depth times)        │   #
#    │   Each layer:                              │   #
#    │   - LayerNorm + Multi-Head Attn + residual │   #
#    │   - LayerNorm + MLP + residual             │   #
#    └────────────────┬───────────────────────────┘   #
#                     │                               #
#     Final sequence: [B,257,hidden_dim]              #
#                     │                               #
#                     v                               #
#          Extract the first token (CLS)              #
#                     │                               #
#                     v                               #
#         Fully-connected layer to actions (4D)       #
#                                                     #
# --------------------------------------------------- #

class ViT(nn.Module):
    # Default parameters for ViT written as class attributes
    encoder_type: str = 'SmallStem'
    patch_size: int = 16
    hidden_dim: int = 64
    num_layers: int = 4
    num_heads: int = 4
    mlp_dim: int = 128
    dropout_rate: float = 0.0
    cnn_channels: tuple = (32, 96, 192, 384)
    action_token_num: int = 1
    use_language_token: bool = False
    fine_tune_pretrained_image_encoder: bool = False
    image_embedding_noise: float = 0.
    use_differential_transformer: bool = False
    return_attention_map: bool = False
    add_positional_embedding: bool = True
    include_class_token: bool = False

    def setup(self):
        if self.encoder_type == 'DINOv2':
            pretrained_model = FlaxDinov2Model.from_pretrained("facebook/dinov2-base")
            self.image_encoder = FlaxDinov2Model(pretrained_model.config).module
        elif self.encoder_type == 'CLIP':
            pretrained_model = FlaxCLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
            self.image_encoder = FlaxCLIPVisionModel(pretrained_model.config).module

    @nn.compact
    def __call__(self, images, instruction_embeddings, train: bool = True, image_embeddings=None):
        # Input instruction_embeddings: [batch, token_length, embedding_dim]
        B, H, W, C = images.shape
        if self.encoder_type == 'EfficientNet':
            assert H == 300 and W == 300, "Input image size must be 300x300."
        elif self.encoder_type in ["DINOv2", "CLIP", "Siglip"]:
            assert H == 224 and W == 224, "Input image size must be 224x224."

        # 1. Encode image into patches
        if self.encoder_type == 'SmallStem':
            patches = SmallStem(
                patch_size=self.patch_size,
                num_features=self.hidden_dim,
                features=self.cnn_channels,
            )(images)
        elif self.encoder_type == 'PatchEncoder':
            patches = PatchEncoder(
                patch_size=self.patch_size,
                num_features=self.hidden_dim,
            )(images)
        elif self.encoder_type == 'EfficientNet':
            config = MODEL_CONFIGS['efficientnet-b3']
            # need to manually normalize the image first, as EfficientNet does not do normalization
            images = normalize_images(images)
            image_embeddings = EfficientNet(config=config)(images, train=train)
            patches = nn.Conv(features=self.hidden_dim, kernel_size=(1, 1))(image_embeddings)
        elif self.encoder_type == 'DINOv2':
            # project image embeddings down to the Transformer input dimension
            raw_images = images / 255.0
            DINO_image_mean = jnp.array([0.485, 0.456, 0.406])
            DINO_image_std = jnp.array([0.229, 0.224, 0.225])
            raw_images = (raw_images - DINO_image_mean[None, None, None]) / DINO_image_std[None, None, None]
            # Flax DINOv2 module treats the last dimension as color, which is different from the pretrained version
            # raw_images = raw_images.transpose(0, 3, 1, 2)
            outputs = self.image_encoder(raw_images, output_attentions=True)
            self.sow('intermediates', 'DINO_attention_map', outputs.attentions)
            if self.include_class_token:
                image_embeddings = outputs.last_hidden_state
            else:
                image_embeddings = outputs.last_hidden_state[:, 1:]
            if self.image_embedding_noise > 0:
                noise_key = self.make_rng('embedding_noise')
                noise = jax.random.normal(noise_key, image_embeddings.shape)
                # do not add embedding noise during evaluation
                image_embeddings = image_embeddings + self.image_embedding_noise * float(train) * noise
            if not self.fine_tune_pretrained_image_encoder:
                image_embeddings = jax.lax.stop_gradient(image_embeddings)
            patches = nn.Dense(
                self.hidden_dim,
                name="image_embedding_projection",
            )(image_embeddings)
        elif self.encoder_type == 'CLIP':
            # project image embeddings down to the Transformer input dimension
            raw_images = images / 255.0
            CLIP_image_mean = jnp.array([0.48145466, 0.4578275, 0.40821073])
            CLIP_image_std = jnp.array([0.26862954, 0.26130258, 0.27577711])
            raw_images = (raw_images - CLIP_image_mean[None, None, None]) / CLIP_image_std[None, None, None]
            # Flax CLIP module treats the last dimension as color, which is different from the pretrained version
            # raw_images = raw_images.transpose(0, 3, 1, 2)
            image_embeddings = self.image_encoder(raw_images).last_hidden_state[:, 1:]
            if not self.fine_tune_pretrained_image_encoder:
                image_embeddings = jax.lax.stop_gradient(image_embeddings)
            patches = nn.Dense(
                self.hidden_dim,
                name="image_embedding_projection",
            )(image_embeddings)
        elif self.encoder_type == 'Siglip':
            patches = nn.Dense(
                self.hidden_dim,
                name="image_embedding_projection",
            )(image_embeddings)
        else:
            raise NotImplementedError("Unknown encoder type for ViT")

        patches = patches.reshape(B, -1, self.hidden_dim)

        if self.use_language_token:
            language_token_num = instruction_embeddings.shape[1]
            # project token embeddings to the same dimension as the image patches
            token_embedding = nn.Dense(
                self.hidden_dim,
                name="language_token_projection"
            )(instruction_embeddings)
            patches = jnp.concatenate([token_embedding, patches], axis=1)

        # ---------------------------
        # Step 2: Add class token
        #
        # FYI: Why do we need the class token?
        # The reason is that pure Transformer processes patch sequences for images
        # without a dedicated token to aggregate semantic information of the whole image.
        # Adding class token is equivalent to adding a special vector at the beginning
        # of the sequence that "represents the entire image". Through self-attention,
        # Transformer allows this class token to interact with all patches,
        # thereby gathering features of the entire image in the class token.
        # cls = self.param('cls', nn.initializers.zeros, (1, 1, self.hidden_dim))
        # cls_tokens = jnp.tile(cls, [B, 1, 1])  # [B, 1, hidden_dim]

        # TODO: do not consider window size for now
        action_tokens = jnp.zeros((B, self.action_token_num, self.hidden_dim))
        x = jnp.concatenate([patches, action_tokens], axis=1)

        # ---------------------------
        # Step 3: Positional Embedding
        #
        # Since Transformer has no position information about the sequence itself,
        # learnable position embeddings need to be added to let the model
        # know the relative positions between patches.
        if self.add_positional_embedding:
            pos_embedding = self.param(
                'pos_embedding',
                nn.initializers.normal(stddev=0.02),
                (1, x.shape[1], self.hidden_dim)
            )
        else:
            pos_embedding = self.param(
                'pos_embedding',
                nn.initializers.normal(stddev=0.02),
                (1, self.action_token_num, self.hidden_dim)
            )
            pos_embedding = jnp.concatenate([jnp.zeros([1, *patches.shape[1:]]), pos_embedding], axis=1)
        x = x + pos_embedding
        x = nn.Dropout(self.dropout_rate)(x, deterministic=not train)

        # ---------------------------
        # Step 4: Stack of Transformer Encoder layers
        attention_mask = jnp.ones((B, 1, x.shape[1], x.shape[1]))
        # language tokens only attend to themselves
        if self.use_language_token:
            attention_mask = attention_mask.at[:, :, :language_token_num, language_token_num:].set(False)
        # language and image tokens do not attend to action tokens
        attention_mask = attention_mask.at[:, :, :-self.action_token_num, -self.action_token_num:].set(False)
        x, attention_map = Transformer(
            embedding_dim=self.hidden_dim,
            num_layers=self.num_layers,
            mlp_dim=self.mlp_dim,
            num_attention_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=0.0,
            use_differential_transformer=self.use_differential_transformer,
            return_attention_map=self.return_attention_map,
        )(x, attention_mask, train=train)

        action_embeddings = x[:, -self.action_token_num:] # [batch_size, action_token_num, hidden_dim]
        return action_embeddings, attention_map