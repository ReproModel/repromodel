import torch
import torch.nn as nn
from repromodel_core.src.decorators import enforce_types_and_ranges, tag

# In tag decorator, specify custom task, subtask, modality, and submodality. 
# If two or more values are needed, add them to the list. 
# For example, submodality=["RGB", "grayscale"].
@tag(task=["classification"], subtask=["binary"], modality=["images"], submodality=["RGB"])
class CustomModel(nn.Module):
    # Specify here every input with:
    # type: required
    # default: optional but helpful to pre-fill the value in the frontend
    # range: optional but helpful as it automatically makes a slider in the frontend
    # options: optional but helpful as it automatically makes a dropdown in the frontend
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