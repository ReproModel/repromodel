import torch
import torch.nn as nn
from ..decorators import enforce_types_and_ranges

class CustomModel(nn.Module):
    @enforce_types_and_ranges({
        'lr': {'type': float, 'default': 0.01, 'range': (0.0001, 1.0)},
        'activation': {'type': str, 'default': 'relu', 'options': ['relu', 'sigmoid', 'tanh']}
    })
    def __init__(self, lr, activation):
        super(CustomModel, self).__init__()
        self.lr = lr
        self.activation = activation
        # Define the layers of the model depending on the activation function perhaps
        pass
    
    def forward(self, x):
        # Define the forward pass using the activation function
        return x

# Example of using the model:
model = CustomModel(lr=0.005, activation='sigmoid')
print("Model learning rate:", model.lr)
print("Model activation function:", model.activation)
