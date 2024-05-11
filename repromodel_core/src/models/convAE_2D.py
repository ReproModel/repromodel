import torch.nn as nn
from ..decorators import enforce_types_and_ranges
from copy import deepcopy

class ConvAE_2D(nn.Module):
    @enforce_types_and_ranges({
        'n_layers': {'type': int, 'range': (1, 10)},
        'input_channels': {'type': int, 'range': (1, 1000)},
        'output_channels': {'type': int, 'range': (1, 1000)},
        'activation': {'type': str, 'options': ['relu', 'sigmoid', 'tanh', 'softmax', 'leaky_relu']},
    })
    def __init__(self, n_layers: int, input_channels: int, output_channels: int, activation: str):
        super(ConvAE_2D, self).__init__()
        self.activation = activation

        # Activation functions dictionary
        activation_functions = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'softmax': nn.Softmax(dim=1),  # Specifying the dimension is important for softmax
            'leaky_relu': nn.LeakyReLU()
        }

        # Validate and set activation function
        if activation not in activation_functions:
            raise ValueError("Unsupported activation function")
        self.activation_func = activation_functions[activation]

        # Encoder
        encoder_layers = []
        in_channels = input_channels
        for i in range(n_layers):
            out_channels = 16 * (2 ** i)
            encoder_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                deepcopy(self.activation_func)
            ])
            in_channels = out_channels
        self.encoder = nn.Sequential(*encoder_layers)

        # Assuming the autoencoder halves the dimension each time and doubles the channels
        in_channels = out_channels  # from the last encoder layer

        # Decoder: reversing the operation of the encoder
        decoder_layers = []
        for i in range(n_layers - 1, -1, -1):
            out_channels = 16 * (2 ** i)
            if i == 0:
                current_out_channels = output_channels  # This should be the original input_channels for typical autoencoders
            else:
                current_out_channels = 16 * (2 ** (i - 1))

            decoder_layers.extend([
                nn.ConvTranspose2d(in_channels, current_out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(current_out_channels),
                deepcopy(self.activation_func)
            ])
            in_channels = current_out_channels  # Update in_channels for the next iteration

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
