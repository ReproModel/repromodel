from ..decorators import enforce_types_and_ranges, tag

# In tag decorator, specify custom task, subtask, modality, and submodality. 
# If two or more values are needed, add them to the list. 
# For example, submodality=["RGB", "grayscale"].
@tag(task=["classification"], subtask=["binary"], modality=["images"], submodality=["grayscale"])
class CustomMetric:
    # Specify here every input with:
    # type: required
    # default: optional but helpful to pre-fill the value in the frontend
    # range: optional but helpful as it automatically makes a slider in the frontend
    # options: optional but helpful as it automatically makes a dropdown in the frontend
    @enforce_types_and_ranges({
        'weight': {'type': float, 'range': (0.0, 1.0)},  
        'smooth': {'type': float, 'range': (0.0, 1.0), 'default': 0.001}  
    })
    def __init__(self, weight, smooth):
        """
        Initialize any attributes or parameters needed for the metric computation.
        
        Args:
            weight (float): A weighting factor for the metric, between 0.0 and 1.0.
            smooth (float): A smoothing factor for the metric, between 0.0 and 1.0.
        """
        self.weight = weight
        self.smooth = smooth
        self.correct = 0
        self.total = 0

    def forward(self, inputs, targets):
        
        # Calculate the score based on the desired method
        # Example:
        score = (inputs - targets) / self.smooth
        return score
