from ..utils import print_to_file

class CustomEarlyStopping:
    def __init__(self, min_lr: float = 1e-6):
        # Common attributes
        self.min_lr = min_lr
        self.best_score = None
        # self.should_stop variable is used to trigger stopping during the training.
        self.should_stop = False
        self.current_epoch = 0

    # Example of the step
    def step(self, current_value, current_lr=None, current_epoch=None):
        if current_epoch is not None:
            self.current_epoch = current_epoch
        
        if self.best_score is None or current_value < self.best_score:
            self.best_score = current_value
            self.should_stop = False
        else:
            if current_lr < self.min_lr:  # Some custom logic to decide to stop
                self.should_stop = True

    # Needed for saving the checkpoint
    def state_dict(self):
        return {'best_score': self.best_score,
                'should_stop': self.should_stop,
                'current_epoch': self.current_epoch}

    # needed for loading the checkpoint
    def load_state_dict(self, state_dict):
        try:
            self.best_score = state_dict['best_score']
            self.should_stop = state_dict['should_stop']
            self.current_epoch = state_dict['current_epoch']
            print_to_file(f'Early Stopping state loaded')
        except KeyError as e:
            print_to_file(f'Error {e} occurred. State_dict not loaded')

