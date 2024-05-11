import torch
from ..decorators import enforce_types_and_ranges

class CustomMetric:
    @enforce_types_and_ranges({
        'weight': {'type': float, 'range': (0.0, 1.0)}  # Assuming weight should be between 0.0 and 1.0
    })
    def __init__(self, weight):
        """
        Initialize any attributes or parameters needed for the metric computation.
        
        Args:
            weight (float): A weighting factor for the metric, between 0.0 and 1.0.
        """
        self.weight = weight
        self.correct = 0
        self.total = 0

    def update(self, outputs, labels):
        """
        Update the state of the metric with results from a new batch.
        
        Args:
            outputs (torch.Tensor): The model's predictions.
            labels (torch.Tensor): The ground truth labels.
        """
        preds = torch.argmax(outputs, dim=1)
        self.correct += (preds == labels).sum().item() * self.weight
        self.total += labels.size(0)

    def compute(self):
        """
        Compute the metric based on updates.
        
        Returns:
            float: The computed metric.
        """
        if self.total == 0:
            return 0.0  # To avoid division by zero
        return self.correct / self.total

    def reset(self):
        """
        Reset the metric state to start computing from scratch.
        """
        self.correct = 0
        self.total = 0
