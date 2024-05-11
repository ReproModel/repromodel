import torch
import torch.nn as nn
from ..decorators import enforce_types_and_ranges

class Conv3dEncoder(nn.Module):
    @enforce_types_and_ranges({
    'n_layers': {'type': int, 'range': (1, 10)},
    'input_channels': {'type': int, 'range': (1, 1000)}
    })
    def __init__(self, n_layers: int, input_channels: int):
        super(Conv3dEncoder, self).__init__()
        # Ensure the number of layers is at least 1
        n_layers = max(1, n_layers)

        # Encoder
        encoder_layers = []
        in_channels = input_channels
        for i in range(n_layers):
            out_channels = 16 * (2 ** i)
            encoder_layers.extend([
                nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU()
            ])
            in_channels = out_channels

        self.encoder = nn.Sequential(*encoder_layers)
        self.out_channels = in_channels  # Number of output channels of the last layer
        
    def forward(self, x):
        return self.encoder(x)

class Conv3dDecoder(nn.Module):
    @enforce_types_and_ranges({
    'n_layers': {'type': int, 'range': (1, 10)},
    'in_channels': {'type': int, 'range': (1, 1024)},
    'output_channels': {'type': int, 'range': (1, 1000)},
    'activation': {'type': str, 'options': ['relu', 'sigmoid']}
    })
    def __init__(self, n_layers: int, in_channels: int, output_channels: int, activation: str):
        super(Conv3dDecoder, self).__init__()
        # Decoder
        decoder_layers = []
        for i in range(n_layers - 1, -1, -1):
            out_channels = 16 * (2 ** i)
            decoder_layers.extend([
                nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU()
            ])
            if i > 0:
                in_channels = out_channels
        # Adjust the last layer to match the output channels
        decoder_layers[-3] = nn.ConvTranspose3d(in_channels, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        if activation == 'sigmoid':
            decoder_layers[-2] = nn.BatchNorm3d(output_channels)
            decoder_layers[-1] = nn.Sigmoid()  # Using Sigmoid activation for the output layer
        else:
            #remove ReLU and BatchNorm
            decoder_layers.pop(-1)
            decoder_layers.pop(-1)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        return self.decoder(x)
        
class Conv3dAutoencoder(nn.Module):
    @enforce_types_and_ranges({
    'n_layers': {'type': int, 'range': (1, 10)},
    'input_channels': {'type': int, 'range': (1, 1000)},
    'output_channels': {'type': int, 'range': (1, 1000)},
    'activation': {'type': str, 'options': ['relu', 'sigmoid']},
    'fusion': {'type': str, 'options': ['intermediate', 'final']}
    })
    def __init__(self, n_layers: int, input_channels: int, output_channels: int, activation: str, fusion: str):
        super(Conv3dAutoencoder, self).__init__()
        self.encoder = Conv3dEncoder(n_layers=n_layers, input_channels=input_channels)
        self.out_channels = self.encoder.out_channels
        self.fusion = fusion
        if self.fusion != 'intermediate':
            self.decoder = Conv3dDecoder(n_layers=n_layers, in_channels=self.encoder.out_channels, output_channels=output_channels, activation=activation)
        
    def forward(self, x):
        x = self.encoder(x)
        if self.fusion != 'intermediate':
            x = self.decoder(x)
        return x