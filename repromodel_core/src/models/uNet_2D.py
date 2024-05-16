import torch
import torch.nn as nn
from ..decorators import enforce_types_and_ranges

# Re-defining the necessary building blocks for 2D UNet
class _ConvBlock(nn.Module):
    @enforce_types_and_ranges({
        'in_channels': {'type': int, 'range': (1, float('inf'))}, 
        'out_channels': {'type': int, 'range': (1, float('inf'))}
    })
    def __init__(self, in_channels, out_channels):
        super(_ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x

class _EncoderBlock(nn.Module):
    @enforce_types_and_ranges({
        'in_channels': {'type': int, 'range': (1, float('inf'))}, 
        'out_channels': {'type': int, 'range': (1, float('inf'))}
    })
    def __init__(self, in_channels, out_channels):
        super(_EncoderBlock, self).__init__()
        self.conv_block = _ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv_block(x)
        skip_connection = x
        x = self.pool(x)
        return skip_connection, x

class _DecoderBlock(nn.Module):
    @enforce_types_and_ranges({
        'in_channels': {'type': int, 'range': (1, float('inf'))}, 
        'out_channels': {'type': int, 'range': (1, float('inf'))}, 
        'skip_channels': {'type': int, 'range': (1, float('inf'))}
    })
    def __init__(self, in_channels, out_channels, skip_channels):
        super(_DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = _ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip_connection):
        x = self.up(x)
        x = torch.cat([x, skip_connection], dim=1)
        x = self.conv_block(x)
        return x

# Separate classes for Encoder and Decoder in 2D U-Net
class _Encoder2D(nn.Module):
    @enforce_types_and_ranges({
        'in_channels': {'type': int, 'range': (1, float('inf'))}, 
        'num_layers': {'type': int, 'range': (1, float('inf'))}
    })
    def __init__(self, in_channels, num_layers=4):
        super(_Encoder2D, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else 2**(4 + i - 1)
            self.out_ch = 2**(4 + i)
            self.layers.append(_EncoderBlock(in_ch, self.out_ch))

    def forward(self, x):
        skip_connections = []
        for layer in self.layers:
            skip_connection, x = layer(x)
            skip_connections.append(skip_connection)
        return skip_connections, x

class _Decoder2D(nn.Module):
    @enforce_types_and_ranges({
        'out_channels': {'type': int, 'range': (1, float('inf'))}, 
        'num_layers': {'type': int, 'range': (1, float('inf'))}, 
        'activation': {'type': str, 'default': 'sigmoid', 'values': ['sigmoid', None]}
    })
    def __init__(self, out_channels, num_layers=4, activation='sigmoid'):
        super(_Decoder2D, self).__init__()
        self.activation = activation
        self.layers = nn.ModuleList()
        for i in reversed(range(num_layers)):
            in_ch = 2**(4 + i + 1)
            out_ch = 2**(4 + i)
            skip_ch = out_ch
            self.layers.append(_DecoderBlock(in_ch, out_ch, skip_ch))
        self.final_layer = nn.Conv2d(16, out_channels, kernel_size=1)
        if activation == 'sigmoid':
            self.activation_layer = nn.Sigmoid(dim=out_channels)
        elif activation == 'softmax':
            self.activation_layer = nn.Softmax(dim=out_channels)

    def forward(self, x, skip_connections):
        for i, layer in enumerate(self.layers):
            x = layer(x, skip_connections[-(i + 1)])
        x = self.final_layer(x)
        if self.activation is not None:
            return self.activation_layer(x)
        else:
            return x

class UNet_2D(nn.Module):
    @enforce_types_and_ranges({
        'in_channels': {'type': int, 'range': (1, float('inf'))}, 
        'out_channels': {'type': int, 'range': (1, float('inf'))}, 
        'num_layers': {'type': int, 'range': (1, float('inf'))}, 
        'activation': {'type': str, 'default': 'sigmoid', 'values': ['sigmoid', None]}
    })
    def __init__(self, in_channels, out_channels, num_layers=4, activation='sigmoid'):
        super(UNet_2D, self).__init__()
        self.encoder = _Encoder2D(in_channels, num_layers)
        self.bottleneck = _ConvBlock(2**(4 + num_layers - 1), 2**(4 + num_layers))
        self.decoder = _Decoder2D(out_channels, num_layers, activation=activation)

    def forward(self, x):
        skip_connections, x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skip_connections)
        return x