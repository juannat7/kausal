import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseObservables

def get_activation(activation):
    """
    Get the activation function by name.

    Parameters:
        activation (str): Name of the activation function.

    Returns:
        Callable: Activation function.
    """
    assert activation in ['relu', 'tanh', 'sigmoid', 'leaky_relu'], f'{activation} is not yet supported!'
    
    return {
        'relu': nn.ReLU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
        'leaky_relu': nn.LeakyReLU()
    }.get(activation)


class CNNFeatures(BaseObservables):
    """
    Generate latent vector z for input data X, with convolution-based model.

    Parameters:
        in_channels (int): Number of input channels.
        hidden_channels (List[int]): List of number of channels in the intermediate layers.
        out_channels (int): Number of output channels.
        activation (str): The type of activation to be used.

    Returns:
        self.
    """

    def __init__(
        self,
        in_channels = 3,
        hidden_channels = [16, 32],
        out_channels = 3,
        activation = 'sigmoid',
        random_seed = 42
    ):
        super().__init__()

        # Automatically assign device based on GPU availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.activation = get_activation(activation)
        self.random_seed = random_seed

        # 1. Construct encoder network
        forward_channels = [in_channels, ] + hidden_channels
        self.encoder = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(forward_channels[i], forward_channels[i + 1], kernel_size=3, stride=2, padding=1, bias=False),
                    self.activation if i < len(hidden_channels) - 1 else nn.Identity()
                )
                for i in range(len(hidden_channels))
            ]
        ).to(self.device)

        # 2. Construct decoder network
        reverse_channels = hidden_channels[::-1] + [out_channels, ]
        self.decoder = nn.Sequential(
            *[
                nn.Sequential(
                    nn.ConvTranspose2d(reverse_channels[i], reverse_channels[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                    self.activation if i < len(hidden_channels) - 1 else nn.Identity()
                )
                for i in range(len(hidden_channels))
            ]
        ).to(self.device)

    def forward(
        self, 
        X
    ):
        """
        Lifting operations for downstream CI evaluation (e.g., shape matching, device transfers, etc).
        Assumes input of shape (C, H, W, T).
        """
        assert X.ndim == 4, '2D convolution assumes input of shape (C, H, W, T)!'
        C, H, W, T = X.shape
        
        X = X.to(self.device)
        X = X.permute(3, 0, 1, 2) # to shape (T, C, H, W)
        
        X = self.encoder(X) # Lifting operation
        
        X = X.reshape(T, -1).permute(1, 0) # to shape (M, T)
        X = X.to('cpu')
        return X

    def fit(
        self,
        x = None,
        y = None,
        epochs = 1000,
        lr = 1e-3,
        batch_size = 32,
        **kwargs
    ):
        """
        Train/fit model.

        Parameters:
            x (torch.Tensor): Input data.
            y (torch.Tensor): Target data.
            epochs (int): Number of training epochs.
            lr (float): Learning rate.
            batch_size (int): Number of batch sizes.
        
        Returns:
            List[float]: Training loss for each epoch.
        """
        if x is None or y is None:
            raise ValueError("Input and target data (x, y) must be provided.")

        x, y = self._transform(x), self._transform(y)

        # Setup dataloader
        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Set up optimizer and loss fn
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()

        # Training loop
        losses = []
        for epoch in range(epochs):
            self.train()
            epoch_loss = 0
    
            for x_batch, y_batch in dataloader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
    
                # Compute loss and backpropagate
                loss = loss_fn(
                    self.decoder(self.encoder(x_batch)), 
                    y_batch
                )
                
                loss.backward()
                optimizer.step()
    
                # Accumulate batch loss
                epoch_loss += loss.item()
    
            # Track average loss per epoch
            losses.append(epoch_loss / len(dataloader))

        return losses

    def _transform(self, x):
        """Assumes input of shape (C, H, W, T), transform to Conv2d-like input of shape (T, H, W, C)."""
        return x.permute(3, 0, 1, 2)
        