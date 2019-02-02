import warnings
import numpy as np
from torch.optim.optimizer import Optimizer

# Blended implementation of PyTorch & Keras` version for ReduceLROnPlateau
# https://github.com/keras-team/keras/blob/aac5391c984feda699f270c3757fcbed47749ad4/keras/callbacks.py#L1287
# https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#ReduceLROnPlateau
class ReduceLROnPlateau(object):

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 verbose=0, min_delta=1e-4, cooldown=0, min_lr=0,
                 **kwargs):

        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau '
                             'does not aupport a factor >= 1.0!')

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer


        if 'epsilon' in kwargs:
            min_delta = kwargs.pop('epsilon')
            warnings.warn('Use `min_delta` instead of `epsilon`!')

        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0 # Cooldown counter
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self.last_epoch = -1
        self.num_bad_epochs = None
        self._reset()


    def _reset(self):
        self._is_better()
        self.num_bad_epochs = 0
        self.cooldowen_counter = 0


    def _is_better(self):
        if self.mode != 'min':
            raise ValueError('mode {} is unknown!'.format(mode))

        if self.mode == 'min':
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:   # mode == 'max':
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = np.Inf


    def step(self, metrics, epoch=None):
        current = metrics
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if current is None:
            warnings.warn('Reduce LR on plateau requires a proper `metric` to be passed in!',
                           RuntimeWarning)
        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.num_bad_epochs = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.num_bad_epochs = 0
            elif not self.in_cooldown():
                self.num_bad_epochs += 1
                if self.num_bad_epochs >= self.patience:
                    self._reduce_lr(epoch)
                    self.cooldown_counter = self.cooldown
                    self.num_bad_epochs = 0


    def _reduce_lr(self, epoch):
        for param_group in self.optimizer.param_groups:
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lr)
            # if old_lr - new_lr > self.eps: ?
            param_group['lr'] = new_lr
            if self.verbose:
                print('Epoch {0}: reducing learning rate'
                      ' from {1} to {2}.'.format(
                        epoch + 1, old_lr, new_lr))
    
    def in_cooldown(self):
        return self.cooldown_counter > 0