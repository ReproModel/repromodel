from .customEarlyStopping import CustomEarlyStopping
from ..decorators import enforce_types_and_ranges
from ..utils import print_to_file
class ScheduledStop(CustomEarlyStopping):
    @enforce_types_and_ranges({
        'max_epochs': {'type': int, 'range': (1, 1000000)}  # Specifying validation rules for max_epochs
    })
    def __init__(self, max_epochs: int):
        super().__init__()
        self.max_epochs = max_epochs - 1  # Adjust for zero-indexing if needed

    def step(self, current_epoch):
        self._handle_scheduled_stopping(current_epoch)

    def _handle_scheduled_stopping(self, current_epoch):
        if current_epoch >= self.max_epochs:
            self.should_stop = True
            print_to_file(f"Early stopping triggered due to reaching the maximum number of epochs: {self.max_epochs + 1}")
