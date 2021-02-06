import torch
import timeit
import numpy as np
#from .train_diagnostics2 import *
from .train_diagnostics import *
from .train_metrics import *
from .jcollections import *
from functools import partial

class train_history:
    def __init__(self, trainer):
        self.trainer = trainer
        self.config = trainer.config
        self.epochs = defaultdict(list)
        self.added_modules = set()
        self.before_batch = []
        self.before_epoch = []
        self.after_epoch = []
        self.after_batch = []
        self.metrics
        for m in self.config.modules:
            self.require_module(m)
        self.config.add_hook('metrics', self.create_modules)
        self.config.add_hook('modules', self.create_modules)
       
    def __del__(self):
        self.config.del_hook('metrics', self.create_modules)
        self.config.del_hook('modules', self.create_modules)

    @property
    def metrics(self):
        try: 
            return self._metrics
        except AttributeError:
            self._metrics = [ m(self) for m in self.config.metrics ]
            for m in self._metrics:
                m.requirements()
            return self._metrics

    def create_modules(self):
        self.require_module(self.config.modules)

    def require_module(self, *modules):
        for module in modules:
            if module.__name__ not in self.added_modules:
                m = module(self)
                if m.requirements():
                    print(f"not using module {module.__name__}")
                else:
                    #print(f"adding module {module.__name__}")
                    if getattr(m, "before_batch", None) != None:
                        self.before_batch.append(m)
                    if getattr(m, "after_batch", None) != None:
                        self.after_batch.append(m)
                    if getattr(m, "before_epoch", None) != None:
                        self._before_epoch.append(m)
                    if getattr(m, "after_epoch", None) != None:
                        self.after_epoch.append(m)
                    self.added_modules.add(module.__name__)

    def create_epoch(self, phase):
        return Epoch(self, phase)

    def register_epoch(self, epoch):
        self.epochs[epoch['phase']].append(epoch)
    
    def plot(self, *metric, **kwargs):
        if len(metric) == 0:
            self.plotf(loss, 'loss', **kwargs)
        else:
            for m in metric:
                m = m(self)
                self.plotf(m, m.__name__, **kwargs)

    def plot_train(self, metric=None, ylabel='loss', start=0, yscale=None, **kwargs):
        if metric is None:
            metric = loss(self)
        x = [ epoch['epoch'] for epoch in self.epochs['train'][start:] ]
        yt = [ metric.value(epoch) for epoch in self.epochs['train'][start:] ]
        plt.figure(**kwargs)
        plt.plot(x, yt, label='train')
        plt.ylabel(ylabel)
        plt.xlabel("epochs")
        if yscale is not None:
            plt.yscale(yscale)
        plt.legend()
        plt.show()

    def plotf(self, metric, ylabel, start=0, yscale=None, **kwargs):
        x = [ epoch['epoch'] for epoch in self.epochs['train'][start:] ]
        yt = [ metric.value(epoch) for epoch in self.epochs['train'][start:] ]
        yv = [ metric.value(epoch) for epoch in self.epochs['valid'][start:] ]
        plt.figure(**kwargs)
        plt.plot(x, yt, label='train')
        plt.plot(x, yv, label='valid')
        plt.ylabel(ylabel)
        plt.xlabel("epochs")
        if yscale is not None:
            plt.yscale(yscale)
        plt.legend()
        plt.show()
        
class Epoch(defaultdict):
    def __init__(self, history, phase):
        super().__init__(float)
        trainer = history.trainer
        self['epoch'] = trainer.epochid
        self['phase'] = phase
        self.history = history
        self.config = history.config
        self.report = False

    def time(self):
        return self['endtime'] - self['starttime']

    def before_epoch(self):
        if self.report:
            self['starttime'] = timeit.default_timer()
            for m in self.history.before_epoch:
                m.before_epoch(self)

    def before_batch(self, X, y):
        if self.report:
            for m in self.history.before_batch:
                m.before_batch(self, X, y)

    def after_batch(self, X, y, y_pred, loss):
        if self.report:
            for m in self.history.after_batch:
                    m.after_batch(self, X, y, y_pred, loss)

    def after_epoch(self):
        if self.report:
            for m in self.history.after_epoch:
                m.after_epoch(self)
            self['endtime'] = timeit.default_timer()

    def __repr__(self):
        return ( self["phase"] + \
            ' '.join([ f' {metric.__name__}: {metric.value(self):.6f}'
                for metric in self.history.metrics ]))

