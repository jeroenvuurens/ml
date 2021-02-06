from __future__ import print_function, with_statement, division
import torch
from tqdm.autonotebook import tqdm
from torch.optim.lr_scheduler import _LRScheduler
import matplotlib.pyplot as plt
from .train_metrics import loss, name_metrics

class trainer_assist:
    def __init__(self, trainer):
        self.trainer = trainer
        self.model = trainer.model
        self.optimizer = trainer.optimizer

class LinearLR(_LRScheduler):
    """Linearly increases the learning rate between two boundaries over a number of
    iterations.
    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_lr (float, optional): the initial learning rate which is the lower
            boundary of the test. Default: 10.
        num_iter (int, optional): the number of iterations over which the test
            occurs. Default: 100.
        last_epoch (int): the index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter
        return [base_lr / 100 + r * (self.end_lr - base_lr / 100) for base_lr in self.base_lrs]
    

class ExponentialLR(_LRScheduler):
    """Exponentially increases the learning rate between two boundaries over a number of
    iterations.
    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_lr (float, optional): the initial learning rate which is the lower
            boundary of the test. Default: 10.
        num_iter (int, optional): the number of iterations over which the test
            occurs. Default: 100.
        last_epoch (int): the index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(ExponentialLR, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        r = (self.last_epoch + 1) / self.num_iter
        r = [base_lr / 100 * (self.end_lr / (base_lr / 100)) ** r for base_lr in self.base_lrs]
        #print(r[0])
        return r
    
    def step2(self, epoch = None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.get_lr()[0]
    
class LRFinder(trainer_assist):
    """Learning rate range test.
    The learning rate range test increases the learning rate in a pre-training run
    between two boundaries in a linear or exponential manner. It provides valuable
    information on how well the network can be trained over a range of learning rates
    and what is the optimal learning rate.
    Arguments:
        model (torch.nn.Module): wrapped model.
        optimizer (torch.optim.Optimizer): wrapped optimizer where the defined learning
            is assumed to be the lower boundary of the range test.
        criterion (torch.nn.Module): wrapped loss function.
        device (str or torch.device, optional): a string ("cpu" or "cuda") with an
            optional ordinal for the device type (e.g. "cuda:X", where is the ordinal).
            Alternatively, can be an object representing the device on which the
            computation will take place. Default: None, uses the same device as `model`.
    Example:
        >>> lr_finder = LRFinder(net, optimizer, criterion, device="cuda")
        >>> lr_finder.range_test(dataloader, end_lr=100, num_iter=100)
    Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
    fastai/lr_find: https://github.com/fastai/fastai
    """

    def __init__(self, trainer, **kwargs):
        trainer_assist.__init__(self, trainer)
        self.criterion = trainer.config.loss
        self.history = {"lr": [], "loss": []}
        self.best_loss = None

        # Save the original state of the model and optimizer so they can be restored if
        # needed

    def range_test( self, end_lr=10, num_iter=100, step_mode="exp", smooth_f=0.05, diverge_th=5):
        """Performs the learning rate range test.
        Arguments:
            train_loader (torch.utils.data.DataLoader): the training set data laoder.
            val_loader (torch.utils.data.DataLoader, optional): if `None` the range test
                will only use the training loss. When given a data loader, the model is
                evaluated after each iteration on that dataset and the evaluation loss
                is used. Note that in this mode the test takes significantly longer but
                generally produces more precise results. Default: None.
            end_lr (float, optional): the maximum learning rate to test. Default: 10.
            num_iter (int, optional): the number of iterations over which the test
                occurs. Default: 100.
            step_mode (str, optional): one of the available learning rate policies,
                linear or exponential ("linear", "exp"). Default: "exp".
            smooth_f (float, optional): the loss smoothing factor within the [0, 1[
                interval. Disabled if set to 0, otherwise the loss is smoothed using
                exponential smoothing. Default: 0.05.
            diverge_th (int, optional): the test is stopped when the loss surpasses the
                threshold:  diverge_th * best_loss. Default: 5.
        """
        
        # Reset test results
        self.history = {"lr": [], "loss": []}
        self.best_loss = None
        self.trainer.commit('lr_finder')

        # Initialize the proper learning rate policy
        if step_mode.lower() == "exp":
            lr_schedule = ExponentialLR(self.optimizer, end_lr, num_iter)
        elif step_mode.lower() == "linear":
            lr_schedule = LinearLR(self.optimizer, end_lr, num_iter)
        else:
            raise ValueError("expected one of (exp, linear), got {}".format(step_mode))

        if smooth_f < 0 or smooth_f >= 1:
            raise ValueError("smooth_f is outside the range [0, 1]")

        # Create an iterator to get data batch by batch
        iterator = iter(self.trainer.train_dl)
        for iteration in tqdm(range(num_iter)):
            # Get a new set of inputs and labels
            try:
                X, y = next(iterator)
            except StopIteration:
                iterator = iter(self.trainer.train_dl)
                X, y = next(iterator)

            # Train on batch and retrieve loss
            loss, pred_y = self.trainer.train_batch(X, y)
            loss = self._validate()

            # Update the learning rate
            lr_schedule.step()
            self.history["lr"].append(lr_schedule.get_lr()[0])

            # Track the best loss and smooth it if smooth_f is specified
            if iteration == 0:
                self.best_loss = loss
            else:
                loss = smooth_f * loss + (1 - smooth_f) * self.history["loss"][-1]

            # Check if the loss has diverged; if it has, stop the test
            self.history["loss"].append(loss)
            if loss > diverge_th * self.best_loss:
                print("Stopping early, the loss has diverged")
                break
        self.trainer.revert('lr_finder')
        print("Learning rate search finished. See the graph with {finder_name}.plot()")

    def _validate(self):
        running_loss = 0
        l = 0
        with torch.set_grad_enabled(False):
            for X, y in self.trainer.valid_dl:
                # Forward pass and loss computation
                loss, y_pred = self.trainer.loss_xy(X, y)
                running_loss += loss.item()
                l += len(X)
        return running_loss / l

    def plot(self, skip_start=10, skip_end=5, log_lr=True):
        """Plots the learning rate range test.
        Arguments:
            skip_start (int, optional): number of batches to trim from the start.
                Default: 10.
            skip_end (int, optional): number of batches to trim from the start.
                Default: 5.
            log_lr (bool, optional): True to plot the learning rate in a logarithmic
                scale; otherwise, plotted in a linear scale. Default: True.
        """

        if skip_start < 0:
            raise ValueError("skip_start cannot be negative")
        if skip_end < 0:
            raise ValueError("skip_end cannot be negative")

        # Get the data to plot from the history dictionary. Also, handle skip_end=0
        # properly so the behaviour is the expected
        lrs = self.history["lr"]
        losses = self.history["loss"]
        if skip_end == 0:
            lrs = lrs[skip_start:]
            losses = losses[skip_start:]
        else:
            lrs = lrs[skip_start:-skip_end]
            losses = losses[skip_start:-skip_end]

        # Plot loss as a function of the learning rate
        plt.plot(lrs, losses)
        if log_lr:
            plt.xscale("log")
        plt.xlabel("Learning rate")
        plt.ylabel("Loss")
        plt.show()

def plot(epochs, *metric, **kwargs):
    if len(metric) == 0:
        plot(epochs, loss, **kwargs)
    else:
        for m in name_metrics(metric):
            plotf(epochs, m, m.__name__, **kwargs)

def plotf(epochs, metric, ylabel, **kwargs):
    x = [ epoch['epoch'] for epoch in epochs['train'] ]
    yt = [ metric(epoch) for epoch in epochs['train'] ]
    yv = [ metric(epoch) for epoch in epochs['valid'] ]
    plt.figure(**kwargs)
    plt.plot(x, yt, label='train')
    plt.plot(x, yv, label='valid')
    plt.ylabel(ylabel)
    plt.xlabel("epochs")
    plt.legend()
    plt.show()
