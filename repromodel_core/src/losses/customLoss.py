import torch
import torch.nn as nn
import torch.nn.functional as F
from ..decorators import enforce_types_and_ranges

class CustomLoss(nn.Module):
    @enforce_types_and_ranges({
        'weight': {'type': float, 'range': (0.1, 1.0)}  # Assuming weight should be between 0.1 and 1.0
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

# Example usage:
# model_output = torch.tensor([...], requires_grad=True)
# true_values = torch.tensor([...])
# loss_fn = CustomLoss(weight=0.5)
# loss = loss_fn(model_output, true_values)
# loss.backward()  # to compute gradients
