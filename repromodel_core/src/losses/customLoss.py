import torch.nn as nn
import torch.nn.functional as F
from ..decorators import enforce_types_and_ranges, tag

# In tag decorator, specify custom task, subtask, modality, and submodality. 
# If two or more values are needed, add them to the list. 
# For example, submodality=["RGB", "grayscale"].
@tag(task=["regression"], subtask=["error"], modality=["tabular"], submodality=["financial"])
class CustomLoss(nn.Module):
    # Specify here every input with:
    # type: required
    # default: optional but helpful to pre-fill the value in the frontend
    # range: optional but helpful as it automatically makes a slider in the frontend
    # options: optional but helpful as it automatically makes a dropdown in the frontend
    @enforce_types_and_ranges({
        'weight': {'type': float, 'range': (0.1, 1.0), 'default': 1.0}  # Assuming weight should be between 0.1 and 1.0
    })
    def __init__(self, weight=1.0):
        super(CustomLoss, self).__init__()
        self.weight = weight

    def forward(self, inputs, targets):
        """
        Calculate the loss between `inputs` and `targets`, adjusted by a weight.

        Args:
            inputs (torch.Tensor): The predictions of the model.
            targets (torch.Tensor): The true values.

        Returns:
            torch.Tensor: The computed weighted loss.
        """
        # Simple mean squared error, weighted
        loss = F.mse_loss(inputs, targets, reduction='none')
        weighted_loss = loss * self.weight
        return weighted_loss.mean()
