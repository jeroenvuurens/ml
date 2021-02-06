from sklearn.metrics import *
import timeit
import copy
import numpy as np
import math
from tqdm import tqdm_notebook as tqdm
from .helper import *
from .train_diagnostics import *
from .train_metrics import *
from .train_history import *
from .jcollections import *
from .helper import *
from functools import partial

class model:
    def __init__(self, data, lr=0.1, report_phases=['train','valid'], report_metrics = None, metrics = [loss, acc], modules=[], **kwargs):
        kwargs.update({'data':data, 'lr':lr, 'report_phases':report_phases, 'metrics':metrics, 'modules':modules})
        assert 'loss' in kwargs
        assert 'metrics' in kwargs
        self.config = config(**kwargs)
        self.epochid = 0
        self.data = data

    def predict(self, x):
        raise NotImplementedError

    def step(self, X, y):
        raise NotImplementedError

    def parameters(self):
        raise NotImplementedError

    @property
    def model(self):
        raise NotImpementedError

    @property
    def history(self):
        try:
            return self._history
        except:
            self._history = train_history(self)
            return self._history

    def epoch(self, X, y, phase):
        epoch = self.history.create_epoch(phase)
        epoch.report = True
        epoch.before_epoch()
        epoch.before_batch(X, y)
        loss, y_pred = self.loss_xy(X, y)
        epoch['loss'] += loss
        epoch['n'] += len(y)
        epoch.after_batch( X, y, y_pred, loss )
        epoch['loss'] /= epoch['n']
        epoch.after_epoch()
        return epoch

    def loss_xy(self, X, y):
        y_pred = self.predict(X)
        return self.config.loss(y, y_pred).sum(), y_pred
       
    def train_loss(self):
        loss, _ = self.loss_xy( self.data.train_X, self.data.train_y )
        return loss

    def valid_loss(self):
        loss, _ = self.loss_xy( self.data.valid_X, self.data.valid_y )
        return loss

    def train(self, epochs=1, report_frequency=None, verbose=True, bar=True, **kwargs):
        delta = kwargs.pop('delta', None)
        if delta is not None:
            epochs = 1e10
            lowest = []
        if report_frequency == None:
            report_frequency = int(round(epochs / 100) * 10)
        if report_frequency < 1:
            report_frequency = 1
        self.config.update( kwargs )
        X = self.data.train_X
        y = self.data.train_y
        epochidspace = math.ceil(math.log10(epochs))
        for i in tqdm(range(epochs), desc='Total', disable=not bar):
            self.step(X, y)
            if i == epochs - 1 or i % report_frequency == 0:
                epoch = self.epoch(X, y, 'train')
                vepoch = self.epoch(self.data.valid_X, self.data.valid_y, 'valid')
                self.history.register_epoch(epoch)
                self.history.register_epoch(vepoch)
                if verbose:
                    print(f'{self.epochid:{epochidspace}d} {epoch.time():.2f}s {epoch} {vepoch}')
                if delta is not None:
                    l = min(lowest)
                    if vepoch['loss'] - l > delta * l:
                        break
                    if len(lowest) > 5:
                        lowest = lowest[1:]
                    lowest.append(vepoch['loss'])
            self.epochid += 1

    def plot(self, *metric, **kwargs):
        self.history.plot(*metric, **kwargs)

    def plot_X(self, X, marker='.', color='red', **kwargs):
        line_X = order_x(X)
        line_y = self.predict(line_X)
        plt.plot(line_X[:, 0], line_y, marker, color=color, **kwargs)

