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
