from __future__ import print_function, with_statement, division
import torch
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
from .train_metrics import loss, name_metrics
from math import log, exp
from functools import partial

def frange(start, end, steps):
    incr = (end - start) / (steps)
    return (start + x * incr for x in range(steps))

def exprange(start, end, steps, **kwargs):
    return (exp(x) for x in frange(log(start), log(end), steps))

def arange(start, end, steps, **kwargs):
    return np.arange(start, end, steps)

def set_lr(trainer):
    def change(value):
        for param_group in trainer.optimizer.param_groups:
            param_group['lr'] = value
    return change

def set_dropouts(dropouts):
     def change(value):
        for d in dropouts:
           d.p = value
     return change

class tuner:
    def __init__(self, trainer, values, param_update, label='parameter', smooth=0.05, diverge=5, **kwargs):
        self.history = {"lr": [], "loss": []}
        self.best_loss = None
        self.label = label
        self.trainer = trainer
        self.values = list(values)
        self.param_update = param_update
        self.range_test(smooth, diverge)

    def reset(self):
        self.trainer.revert('tuner')

    def next_train(self):
        try:
            return next(self.train_iterator)
        except (StopIteration, AttributeError):
            self.train_iterator = iter(self.trainer.train_dl)
            return next(self.train_iterator)

    def _validate(self):
        running_loss = 0
        l = 0
        with torch.set_grad_enabled(False):
            for X, y in self.trainer.valid_dl:
                loss, y_pred = self.trainer.loss_xy(X, y)
                running_loss += loss.item()
                l += len(X)
        return running_loss / l
 
    def range_test( self, smooth, diverge):
        self.history = {"value": [], "loss": [], "sloss": []}
        self.best_loss = None
        self.trainer.commit('tuner')

        for i in tqdm(self.values):
            self.history["value"].append(i)
            self.param_update(i)
            X, y = self.next_train()

            loss, pred_y = self.trainer.train_batch(X, y)
            loss = self._validate()
            self.history["loss"].append(loss)

            # Track the best loss and smooth it if smooth_f is specified
            try:
                loss = smooth * loss + (1 - smooth) * self.history["loss"][-1]
            except: pass
            self.history["sloss"].append(loss)

            try:
                self.best_loss = min(self.best_loss, loss)
            except:
                self.best_loss = loss

            # Check if the loss has diverged; if it has, stop the test
            if loss > diverge * self.best_loss:
                print("Stopping early, the loss has diverged")
                break
        self.reset()
        print("Learning rate search finished. See the graph with {finder_name}.plot()")

    def plot(self, skip_start=10, smooth=1, log=True):
        if skip_start < 0:
            raise ValueError("skip_start cannot be negative")

        # Get the data to plot from the history dictionary. Also, handle skip_end=0
        # properly so the behaviour is the expected
        values = self.history['value']
        losses = list(self.history['loss'])
        for i, a in enumerate(losses):
            if i > 0:
                losses[i] = smooth * a + (1 - smooth) * losses[i-1]
        m = min(losses)
        imin = next((x[0] for x in enumerate(losses) if x[1] == m))
        skip_end = next((x[0] for x in enumerate(losses) if x[0] > imin and x[1] > m * 5), len(losses))
        values = values[skip_start:skip_end]
        losses = losses[skip_start:skip_end]

        # Plot loss as a function of the learning rate
        plt.plot(values, losses)
        if log:
            plt.xscale("log")
        plt.xlabel(self.label)
        plt.ylabel("Loss")
        plt.show()

#def plot(epochs, *metric, **kwargs):
#    if len(metric) == 0:
#        plot(epochs, loss, **kwargs)
#    else:
#        for m in name_metrics(metric):
#            plotf(epochs, m, m.__name__, **kwargs)

#def plotf(epochs, metric, ylabel, **kwargs):
#    x = [ epoch['epoch'] for epoch in epochs['train'] ]
#    yt = [ metric(epoch) for epoch in epochs['train'] ]
#    yv = [ metric(epoch) for epoch in epochs['valid'] ]
#    plt.figure(**kwargs)
#    plt.plot(x, yt, label='train')
#    plt.plot(x, yv, label='valid')
#    plt.ylabel(ylabel)
#    plt.xlabel("epochs")
#    plt.legend()
#    plt.show()
