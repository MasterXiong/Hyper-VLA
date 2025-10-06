import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Optional, Tuple, Callable # Added Callable for typing

# --- 1. Custom Multi-Head Attention Layer ---
class CustomMultiHeadDotProductAttention(nn.Module):
    """Custom multi-head dot-product attention layer that returns attention weights.

    Attributes:
      num_heads: Number of attention heads.
      qkv_features: Dimension of query, key, value features. If None, defaults to input feature dimension.
      out_features: Output feature dimension. If None, defaults to input feature dimension.
      kernel_init: Weight initializer for projection layers.
      bias_init: Bias initializer for projection layers.
      use_bias: Whether to use bias in projection layers.
      broadcast_dropout: Whether to use the same dropout mask across all heads.
      dropout_rate: Dropout probability for attention weights.
      precision: Precision for einsum operations (None, 'fastest', 'high', 'highest').
      dtype: Data type for computations.
    """
    num_heads: int
    qkv_features: Optional[int] = None
    out_features: Optional[int] = None
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()
    use_bias: bool = True
    broadcast_dropout: bool = True
    dropout_rate: float = 0.0
    precision: Optional[jax.lax.Precision] = None
    dtype: Optional[jnp.dtype] = None

    @nn.compact
    def __call__(self,
                 inputs_q: jax.Array,
                 inputs_kv: Optional[jax.Array] = None,
                 mask: Optional[jax.Array] = None,
                 deterministic: Optional[bool] = None) -> Tuple[jax.Array, jax.Array]:
        """Applies multi-head dot product attention.

        Args:
          inputs_q: Query inputs with shape `[batch_sizes..., length, features]`.
          inputs_kv: Key/value inputs with shape `[batch_sizes..., length, features]`.
                     If None, inputs_q is used as both key and value.
          mask: Attention mask with shape `[batch_sizes..., num_heads, query_length, key_value_length]`.
                Attention weights at positions where mask is False will be ignored.
          deterministic: If False, applies dropout to attention weights; if True, no dropout is applied.

        Returns:
          A tuple (output, attention_weights):
            output: Output of the attention layer with shape `[batch_sizes..., length, features]`.
            attention_weights: Computed attention weights with 
                               shape `[batch_sizes..., num_heads, query_length, key_value_length]`.
        """
        if inputs_kv is None:
            inputs_kv = inputs_q

        features = self.out_features or inputs_q.shape[-1]
        qkv_f = self.qkv_features or inputs_q.shape[-1]

        if qkv_f % self.num_heads != 0:
            raise ValueError(
                f'qkv_features ({qkv_f}) must be divisible by num_heads ({self.num_heads}).'
            )
        head_dim = qkv_f // self.num_heads

        dense_general = nn.DenseGeneral

        # --- Q, K, V Projections ---
        query = dense_general(features=(self.num_heads, head_dim),
                              axis=-1, kernel_init=self.kernel_init, bias_init=self.bias_init,
                              use_bias=self.use_bias, dtype=self.dtype, precision=self.precision,
                              name='query')(inputs_q)
        key = dense_general(features=(self.num_heads, head_dim),
                            axis=-1, kernel_init=self.kernel_init, bias_init=self.bias_init,
                            use_bias=self.use_bias, dtype=self.dtype, precision=self.precision,
                            name='key')(inputs_kv)
        value = dense_general(features=(self.num_heads, head_dim),
                              axis=-1, kernel_init=self.kernel_init, bias_init=self.bias_init,
                              use_bias=self.use_bias, dtype=self.dtype, precision=self.precision,
                              name='value')(inputs_kv)

        dropout_rng = None
        if not deterministic and self.dropout_rate > 0.0:
            dropout_rng = self.make_rng('dropout')

        # --- Calculate Attention Weights ---
        attn_weights = nn.dot_product_attention_weights(
            query,
            key,
            bias=mask,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout_rate,
            broadcast_dropout=self.broadcast_dropout,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=self.precision
        )

        # --- Calculate Output Using Attention Weights ---
        attn_output = jnp.einsum('...hqk,...khd->...qhd', attn_weights, value, precision=self.precision)

        # --- Output Projection ---
        output = dense_general(features=features, axis=(-2, -1),
                               kernel_init=self.kernel_init, bias_init=self.bias_init,
                               use_bias=self.use_bias, dtype=self.dtype, precision=self.precision,
                               name='out')(attn_output)

        return output, attn_weights # Explicitly return output and attention_weights
