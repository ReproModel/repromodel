from .customEarlyStopping import CustomEarlyStopping
from ..decorators import enforce_types_and_ranges

class NoImprovementAfter(CustomEarlyStopping):
    @enforce_types_and_ranges({
    'patience': {'type': int, 'default': 10, 'range': (1, 50)},  # Patience should be a reasonable number of epochs
    'delta': {'type': float, 'default': 0.001, 'range': (0.0, 1.0)},  # Delta should be a small positive float
    'min_lr': {'type': float, 'default':1e-6, 'range': (0.0, 0.000001)}  # Minimum learning rate should be a small positive float
    })
    def __init__(self, patience: int, delta: float, min_lr: float):
        super().__init__()
        self.monitor = 'val_loss'
        self.patience = patience
        self.delta = delta
        self.min_lr = min_lr

        self.epochs_no_improve = 0

    def step(self, current_value, current_lr=None, current_epoch=None):
        self._handle_no_improvement(current_value, current_lr)

    def _handle_no_improvement(self, current_value, current_lr):
        score = -current_value if self.monitor == 'val_loss' else current_value

        if self.best_score is None:
            self.best_score = score
        else:
            if score < self.best_score + self.delta:
                self.epochs_no_improve += 1
            else:
                self.best_score = score
                self.epochs_no_improve = 0

            if(current_lr <= self.min_lr):
                self.should_stop = True
                print(f"Early stopping triggered due to minimum Learning Rate of {self.min_lr} reached.")

            if self.epochs_no_improve >= self.patience:
                self.should_stop = True
                print(f"Early stopping triggered due to no improvement in {self.patience} epochs.")
