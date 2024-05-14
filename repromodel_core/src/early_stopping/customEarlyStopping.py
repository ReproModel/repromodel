from ..decorators import enforce_types_and_ranges
from ..utils import print_to_file
class CustomEarlyStopping:
    def __init__(self):
        # Common attributes
        self.best_score = None
        self.should_stop = False
        self.current_epoch = 0

    def step(self, current_value, current_lr=None, current_epoch=None):
        if current_epoch is not None:
            self.current_epoch = current_epoch
        
        if self.best_score is None or current_value < self.best_score:
            self.best_score = current_value
            self.should_stop = False
        else:
            if current_lr < 1e-4:  # Some custom logic to decide to stop
                self.should_stop = True

    def is_stop(self):
        return self.should_stop

    def state_dict(self):
        return {'best_score': self.best_score,
                'should_stop': self.should_stop,
                'current_epoch': self.current_epoch}

    def load_state_dict(self, state_dict):
        try:
            self.best_score = state_dict['best_score']
            self.should_stop = state_dict['should_stop']
            self.current_epoch = state_dict['current_epoch']
        except KeyError as e:
            print_to_file(f'Error {e} occurred. State_dict not loaded')

