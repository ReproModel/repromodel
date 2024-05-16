import torch
from ..decorators import enforce_types_and_ranges

class AccuracyScore(torch.nn.Module):
    @enforce_types_and_ranges({
        'threshold': {'type': float, 'default': 0.5, 'range': (0.0, 1.0)}  # Threshold should be between 0 and 1
    })
    def __init__(self, threshold=0.5):
        super(AccuracyScore, self).__init__()
        self.threshold = threshold

    def forward(self, inputs, targets):
        if inputs.is_floating_point():
            inputs = (inputs > self.threshold).float()
        
        inputs_flat = inputs.view(inputs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        
        correct_predictions = (inputs_flat == targets_flat).float().sum(1)
        total_predictions = targets_flat.size(1)
        
        accuracy_score = correct_predictions / total_predictions
        
        return accuracy_score.mean()