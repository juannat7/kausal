import jax
import jax.nn
import equinox as eqx

class MLP(eqx.Module):
    layers: list[eqx.nn.Linear]

    def __init__(self, dims, key=jax.random.PRNGKey(42)):
        """
        Multilayer perceptron.

        Parameters:
            dims: All dimensions.
            key: PRNG key for initialization.
        """
        keys = jax.random.split(key, len(dims))
        self.layers = [
            eqx.nn.Linear(dims[i], dims[i + 1], use_bias=False, key=keys[i])
            for i in range(len(dims) - 1)
        ]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x)


class NNTransformModel(eqx.Module):
    encoder: MLP
    decoder: MLP

    def __init__(self, input_dim, hidden_dims, output_dim, key=jax.random.PRNGKey(42)):
        """
        Autoencoder-based lifting model.

        Parameters:
            input_dim: Input dimension of the encoder.
            hidden_dims: List of hidden layer sizes for both encoder and decoder.
            output_dim: Output dimension of the decoder.
            key: PRNG key for initialization.
        """
        encoder_dims = [input_dim] + hidden_dims 
        decoder_dims = hidden_dims[::-1] + [output_dim]
        
        keys = jax.random.split(key, 2)
        self.encoder = MLP(encoder_dims, keys[0])
        self.decoder = MLP(decoder_dims, keys[1])


class CNNEncoder(eqx.Module):
    layers: list[eqx.nn.Conv2d]

    def __init__(self, dims, key=jax.random.PRNGKey(42)):
        """
        A single CNN that performs 2x spatial reduction at each layer.

        Parameters:
            dims: All dimensions.
            key: PRNG key for initialization.
        """
        keys = jax.random.split(key, len(dims))
        self.layers = [
            eqx.nn.Conv2d(
                in_channels=dims[i], out_channels=dims[i + 1], use_bias=False, key=keys[i],
                kernel_size=3,
                stride=2,
                padding=1
            )
            
            for i in range(len(dims) - 1)
        ]

    def __call__(self, x):
        
        if len(x.shape) == 4:
            x = x[..., 0]
            
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x)


class CNNDecoder(eqx.Module):
    layers: list[eqx.Module]

    def __init__(self, dims, key=jax.random.PRNGKey(42)):
        """
        A simple CNN Decoder with alternating transposed convolutions (upsampling) and convolutions (smoothing).
        
        Parameters:
            dims: A list of dimensions for the decoder layers (in reverse order of encoder layers).
            key: PRNG key for initialization.
        """
        keys = jax.random.split(key, 2 * len(dims))  # Separate keys for each layer
        self.layers = [
            layer
            for i in range(len(dims) - 1)
            for layer in [
                # Transposed convolution for upsampling
                eqx.nn.ConvTranspose2d(
                    in_channels=dims[i], out_channels=dims[i + 1], use_bias=False, key=keys[2*i],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1
                ),
                # Convolution for smoothing
                eqx.nn.Conv2d(
                    in_channels=dims[i + 1],
                    out_channels=dims[i + 1],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    use_bias=False,
                    key=keys[2*i + 1]
                )
            ]
        ]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x)


class CNNTransformModel(eqx.Module):
    encoder: CNNEncoder
    decoder: CNNDecoder

    def __init__(self, input_dim, hidden_dims, output_dim, key=jax.random.PRNGKey(42)):
        """
        Autoencoder-based lifting model.

        Parameters:
            input_dim: Input dimension of the encoder.
            hidden_dims: List of hidden layer sizes for both encoder and decoder.
            output_dim: Output dimension of the decoder.
            key: PRNG key for initialization.
        """
        encoder_dims = [input_dim] + hidden_dims 
        decoder_dims = hidden_dims[::-1] + [output_dim]
        
        keys = jax.random.split(key, 2)
        self.encoder = CNNEncoder(encoder_dims, keys[0])
        self.decoder = CNNDecoder(decoder_dims, keys[1])
