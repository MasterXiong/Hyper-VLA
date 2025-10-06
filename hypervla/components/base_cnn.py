from flax import linen as nn
from octo.model.components.vit_encoders import *


class CNN(nn.Module):
    """
    CNN + MLP architecture in JAX/Flax.
    Input: 256 x 256 x 3 image (NHWC)
    Output: 4D action in [-1,1].
    """
    kernel_sizes: tuple = (3, 3, 3, 3)
    strides: tuple = (2, 2, 2, 2)
    features: tuple = (32, 64, 128, 256)
    padding: tuple = (1, 1, 1, 1)
    mlp_hidden_sizes: tuple = (32, 32)
    output_dim: int = 4

    @nn.compact
    def __call__(self, x):
        # normalize images
        x = normalize_images(x, 'default')
        # Convolution layers
        for n, (kernel_size, stride, features, padding) in enumerate(
            zip(
                self.kernel_sizes,
                self.strides,
                self.features,
                self.padding,
            )
        ):
            x = StdConv(
                features=features,
                kernel_size=(kernel_size, kernel_size),
                strides=(stride, stride),
                padding=padding,
            )(x)
            x = nn.GroupNorm()(x)
            x = nn.relu(x)
        # Flatten
        x = x.reshape((x.shape[0], -1))
        # linear layers
        for hidden_size in self.mlp_hidden_sizes:
            x = nn.Dense(
                hidden_size,
            )(x)
            x = nn.relu(x)
        # output layer
        x = nn.Dense(
            self.output_dim,
        )(x)
        # x = nn.tanh(x) # TODO: use tanh or not?
        return x