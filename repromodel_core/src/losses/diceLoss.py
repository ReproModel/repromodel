import torch
from ..decorators import enforce_types_and_ranges

class DiceLoss(torch.nn.Module):
    @enforce_types_and_ranges({
    'smooth': {'type': float, 'default': 1, 'range': (1e-5, 1.0)}  # 'smooth' should be a small positive number
    })
    def __init__(self, smooth: float):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Uncomment if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + self.smooth)/(inputs.sum() + targets.sum() + self.smooth)  
        
        return 1 - dice