import jax
import jax.numpy as jnp
import math
from flax import linen as nn
from typing import Optional

"""
Below is a Flax implementation of "Differential Attention" inspired by the Differential Transformer approach.

-----------------------
Conceptual Explanation
-----------------------
In a standard multi-head attention (MHA) mechanism, we typically have queries (Q), keys (K), and values (V).
For each head, we compute weights via softmax(QK^T) and then multiply by V. This yields a weighted sum of values
that represent contextualized embeddings for each token.

Differential Attention introduces a slight modification to the standard attention mechanism: it uses two sets of 
queries and keys (Q1,K1 and Q2,K2), computing two separate attention distributions:
    
    A1 = softmax(Q1 K1^T)
    A2 = softmax(Q2 K2^T)

Instead of using a single distribution to weight V, it constructs a "differential" distribution by subtracting 
a scaled version of the second distribution A2 from the first A1:

    A = A1 - λ * A2

Here, λ (lambda) is a learnable scalar (or vector) parameter that re-parameterizes how much the second attention 
distribution influences the first. Intuitively:
- A1 can be seen as the "positive" attention focusing on relevant parts of the sequence.
- A2 can be seen as a contrasting distribution that "subtracts out" less relevant information or noise.
- By controlling λ, the model dynamically balances between the original attention and the "negative" one.

After computing A, we use it to weigh V and produce the final output. Further, we apply a form of normalization 
(RMSNorm) and then scale by (1 - λ_init) to maintain stability.

----------------------------------------
Frameworks of the Differential Attention
----------------------------------------
1. The parameter setup and how λ (lambda) is formed:
   - Four sets of parameters: lambda_q1, lambda_k1, lambda_q2, lambda_k2 are learned.
   - We combine them by summing their elementwise products and taking exponentials:
        lambda_1 = exp(sum(lambda_q1 * lambda_k1))
        lambda_2 = exp(sum(lambda_q2 * lambda_k2))
     This yields a "re-parameterized" lambda_full:
        lambda_full = lambda_1 - lambda_2 + lambda_init
   - The attention weights are reshaped and the difference operation (A = A1 - lambda_full * A2) is performed.
   - Then RMSNorm is applied, followed by multiplication by (1 - lambda_init), and finally a linear projection.

2. The conceptual architecture:
   - Input X is projected into Q and K split into (Q1,Q2) and (K1,K2), and V is similarly shaped.
   - Compute A1 = softmax(Q1K1^T), A2 = softmax(Q2K2^T)
   - Construct differential attention: (A1 - λ * A2) @ V
   - Apply GroupNorm (or RMSNorm as in our code) and scale by (1 - λ_init)
   - Finally, concatenate heads and pass through a linear layer.
   
In the code below, we follow these steps in Flax:
- We define a `DifferentialAttention` module that:
  - Projects input embeddings into Q,K,V.
  - Splits Q,K into Q1,Q2 and K1,K2, and arranges V similarly.
  - Computes softmax(Q1K1^T) and softmax(Q2K2^T).
  - Re-parameterizes λ using learned parameters lambda_q1, lambda_k1, lambda_q2, lambda_k2.
  - Takes the difference of attention distributions: A = A1 - λ * A2.
  - Applies attention to V, then uses RMSNorm and scaling.
  - Merges everything back into the original embedding dimension.

This approach potentially reduces noise and helps the attention mechanism focus on more salient parts of the 
sequence.

------------------------
Code Implementation
------------------------
"""

def lambda_init_fn(depth):
    # Initializes lambda based on the depth of the layer.
    # Suggested in the original code: 
    # lambda_init = 0.8 - 0.6 * exp(-0.3 * depth)
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class RMSNorm(nn.Module):
    dim: int
    eps: float = 1e-6
    elementwise_affine: bool = True

    @nn.compact
    def __call__(self, x):
        # Root Mean Square normalization:
        # x_normed = x / sqrt(mean(x^2) + eps)
        normed = x * jax.lax.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)
        if self.elementwise_affine:
            # If affine, we have a learnable weight per dimension
            weight = self.param("weight", nn.initializers.ones, (self.dim,))
            normed = normed * weight
        return normed


class DifferentialAttention(nn.Module):
    """
    Implementation of Differential Attention in Flax.

    Attributes:
    - embed_dim: total embedding dimension of the model.
    - num_heads: number of heads for DiffAttn (note that total Q heads = 2 * num_heads)
      this param should be half of the baseline head number
    - num_kv_heads: number of K,V heads (if not set, defaults to num_heads)
    - depth: layer depth, used for lambda_init calculation.
    """
    embed_dim: int
    num_heads: int
    num_kv_heads: Optional[int] = None
    depth: int = 0
    eps: float = 1e-5

    def __post_init__(self):
        # Flax requires calling super().__post_init__()
        super().__post_init__()

        # If num_kv_heads is not provided, set it equal to num_heads
        if self.num_kv_heads is None:
            object.__setattr__(self, 'num_kv_heads', self.num_heads)

        # n_rep: how many times we need to replicate K,V heads to match Q heads
        # Because DiffAttn uses Q1,Q2 heads, effectively doubling Q heads count, 
        # we maintain a ratio with kv heads.
        n_rep = self.num_heads // self.num_kv_heads
        object.__setattr__(self, 'n_rep', n_rep)

        # head_dim: each head dimension is embed_dim / (2*num_heads) 
        # because Q is [batch, seq_len, 2*num_heads, head_dim] = embed_dim total
        head_dim = self.embed_dim // (2 * self.num_heads)
        object.__setattr__(self, 'head_dim', head_dim)

    def setup(self):
        # Define linear projections for Q, K, V
        # Q: project into embed_dim
        # K,V: project into embed_dim//n_rep so we can reshape into [2*num_kv_heads, head_dim]
        self.q_proj = nn.Dense(self.embed_dim, use_bias=False)
        self.k_proj = nn.Dense(self.embed_dim // self.n_rep, use_bias=False)
        self.v_proj = nn.Dense(self.embed_dim // self.n_rep, use_bias=False)
        self.out_proj = nn.Dense(self.embed_dim, use_bias=False)

        # Initialize lambda_init based on the depth
        self.lambda_init = lambda_init_fn(self.depth)

        # Initialize lambda parameters with a normal distribution
        init_std = 0.1
        self.lambda_q1 = self.param("lambda_q1", nn.initializers.normal(init_std), (self.head_dim,))
        self.lambda_k1 = self.param("lambda_k1", nn.initializers.normal(init_std), (self.head_dim,))
        self.lambda_q2 = self.param("lambda_q2", nn.initializers.normal(init_std), (self.head_dim,))
        self.lambda_k2 = self.param("lambda_k2", nn.initializers.normal(init_std), (self.head_dim,))

        # RMSNorm on concatenated heads dimension: [2 * head_dim]
        # After attention we get shape: [b,h,t,2*head_dim], we normalize across the last dimension.
        self.subln = RMSNorm(2 * self.head_dim, eps=self.eps, elementwise_affine=True)

    def __call__(self, x, attn_mask=None):
        """
        Forward pass of Differential Attention.

        Inputs:
        x: [batch, seq_len, embed_dim] 
           The input sequence embeddings.
        attn_mask: Optional mask for attention [1,1,seq_len,seq_len], e.g. for causal masking.

        Steps:
        1. Project x into Q, K, V.
        2. Reshape Q: [b,t,2*num_heads,head_dim], then split into Q1,Q2
        3. Reshape K: [b,s,2*num_kv_heads,head_dim], similarly split into K1,K2
        4. Reshape V: [b,s,num_kv_heads,2*head_dim]
        5. If n_rep>1, repeat kv heads to match Q heads.
        6. Compute A1 = softmax(Q1K1^T) and A2 = softmax(Q2K2^T)
        7. Compute lambda_full = (exp(sum(lambda_q1*lambda_k1)) - exp(sum(lambda_q2*lambda_k2)) + lambda_init)
        8. Compute final attention A = A1 - lambda_full * A2
        9. Multiply A by V: attn = A @ V
        10. Apply RMSNorm and scale by (1 - lambda_init)
        11. Merge heads and apply out_proj to return [b, t, embed_dim]

        Returns:
        attn_out: [batch, seq_len, embed_dim]
        """
        bsz, tgt_len, _ = x.shape
        src_len = tgt_len

        # Compute Q,K,V
        q = self.q_proj(x)  # [b,t,embed_dim]
        k = self.k_proj(x)  # [b,s,embed_dim//n_rep]
        v = self.v_proj(x)  # [b,s,embed_dim//n_rep]

        # Reshape Q: [b,t,2*num_heads,head_dim]
        q = q.reshape(bsz, tgt_len, 2 * self.num_heads, self.head_dim)
        # Reshape K: [b,s,2*num_kv_heads,head_dim]
        k = k.reshape(bsz, src_len, 2 * self.num_kv_heads, self.head_dim)
        # Reshape V: [b,s,num_kv_heads,2*head_dim]
        v = v.reshape(bsz, src_len, self.num_kv_heads, 2 * self.head_dim)

        # Split Q into Q1,Q2 and K into K1,K2
        q = q.reshape(bsz, tgt_len, self.num_heads, 2, self.head_dim)
        q1 = q[:, :, :, 0, :]
        q2 = q[:, :, :, 1, :]

        k = k.reshape(bsz, src_len, self.num_kv_heads, 2, self.head_dim)
        k1 = k[:, :, :, 0, :]
        k2 = k[:, :, :, 1, :]

        # If n_rep > 1, replicate kv heads
        if self.n_rep > 1:
            k1 = jnp.repeat(k1, self.n_rep, axis=2)
            k2 = jnp.repeat(k2, self.n_rep, axis=2)
            v = jnp.repeat(v, self.n_rep, axis=2)

        # Compute attention scores
        # Q1K1^T: [b,h,t,s], Q2K2^T: [b,h,t,s]
        q1k1 = jnp.einsum('bthd,bshd->bhts', q1, k1)
        q2k2 = jnp.einsum('bthd,bshd->bhts', q2, k2)

        # Add mask if provided
        if attn_mask is not None:
            q1k1 = q1k1 + attn_mask
            q2k2 = q2k2 + attn_mask

        A1 = nn.softmax(q1k1, axis=-1)
        A2 = nn.softmax(q2k2, axis=-1)

        # Compute lambda_full
        lambda_1 = jnp.exp(jnp.sum(self.lambda_q1 * self.lambda_k1))
        lambda_2 = jnp.exp(jnp.sum(self.lambda_q2 * self.lambda_k2))
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        # ----------------------- #
        # Differential attention  #
        # ----------------------- #
        A = A1 - lambda_full * A2
        # ----------------------- #
        # Differential attention  #
        # ----------------------- #

        # Apply attention weights to V
        v = v.transpose(0, 2, 1, 3)  # [b,h,s,2*d]
        attn = jnp.einsum('bhts,bhsd->bhtd', A, v) # [b,h,t,2*d]

        # Apply RMSNorm and scale by (1 - lambda_init)
        attn_reshaped = attn.reshape(bsz * self.num_heads * tgt_len, 2 * self.head_dim)
        attn_normed = self.subln(attn_reshaped)
        attn_normed = attn_normed.reshape(bsz, self.num_heads, tgt_len, 2 * self.head_dim)
        attn_normed = attn_normed * (1.0 - self.lambda_init)

        # Merge heads and output projection
        attn_out = attn_normed.transpose(0, 2, 1, 3).reshape(bsz, tgt_len, self.embed_dim)
        attn_out = self.out_proj(attn_out)
        return attn_out, A



# Simple test to confirm shape
if __name__ == "__main__":
    diff_attn_layer = DifferentialAttention(embed_dim=512, num_heads=8, depth=1)
    x = jnp.ones((2, 10, 512))
    # Create a causal mask [1,1,10,10], upper-triangular with -inf
    causal_mask = jnp.triu(jnp.full((1,1,10,10), -jnp.inf), 1)

    params = diff_attn_layer.init(jax.random.PRNGKey(0), x, attn_mask=causal_mask)
    y = diff_attn_layer.apply(params, x, attn_mask=causal_mask)
    print(y.shape)  # Expected output: [2, 10, 512]
