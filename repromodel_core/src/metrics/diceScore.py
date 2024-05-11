import torch
from ..decorators import enforce_types_and_ranges

class DiceScore(torch.nn.Module):
    @enforce_types_and_ranges({
    'threshold': {'type': float, 'default': 0.5, 'range': (0.0, 1.0)},  # Threshold should be between 0 and 1
    'smooth': {'type': float, 'default': 1, 'range': (1e-5, 1)}   # Smooth value should be a small positive number
    })
    def __init__(self, threshold=0.5, smooth=1e-6):
        super(DiceScore, self).__init__()
        self.threshold = threshold
        self.smooth = smooth

    def forward(self, inputs, targets):
        if inputs.is_floating_point():
            inputs = (inputs > self.threshold).float()
        
        inputs_flat = inputs.view(inputs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        
        intersection = (inputs_flat * targets_flat).sum(1)
        union = inputs_flat.sum(1) + targets_flat.sum(1)
        
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        
        return dice_score.mean()
