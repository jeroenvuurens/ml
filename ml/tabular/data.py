import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from ..kernel.helper import *
from torch.utils.data import TensorDataset 
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from sklearn.model_selection import train_test_split

class TabularDatabunch:
    def __init__(self, config, train, test):
        self.train_ds = train
        self.valid_ds = test
        self.KEYS_DL = {'batch_size', 'shuffle', 'sampler', 'batch_sampler', 'num_workers', 'collate_fn', 'pin_memory', 'drop_last', 'timeout', 'worker_init_fn'}

        config['batch_size'] = min(config.batch_size, len(train))
        self.config = config
        self.batch_size = config.batch_size
        config.add_hook(self.reset, *self.KEYS_DL)
        #self.device = config.device

    def __del__(self):
        self.config.del_hook(self.reset, *self.KEYS_DL)

    @classmethod
    def from_list( cls, l, config ):
        cls.turnoff_multiprocessing_cuda(config)
        tensor = np.matrix(l, dtype=np.float32)
        x = torch.from_numpy(tensor[:, :-1])
        x = x.to(config.device)
        y = torch.from_numpy(tensor[:, -1])
        y = y.to(config.device)
        t = TensorDataset(x, y)
        r = cls(config, t, t)
        return r

    @classmethod
    def from_pd( cls, df, config, valid_perc ):
        cls.turnoff_multiprocessing_cuda(config)
        x = df.iloc[:, :-1]
        y = df.iloc[:, -1:]
        tensors = train_test_split(x, y, test_size = valid_perc, random_state = 0)
        tensors = [ torch.tensor(t.values.astype(np.float32)).to(config.device) for t in tensors ]
        r = cls(config, TensorDataset(tensors[0], tensors[2]), TensorDataset(tensors[1], tensors[3]))
        return r

    @classmethod
    def turnoff_multiprocessing_cuda(cls, config):
        if config.device.type == 'cuda':
            config['pin_memory'] = False
            config['num_workers'] = 0
            #config['shuffle'] = False

    @property
    def device(self):
        return self.config.device

    def reset(self):
        try:
            del self.valid_dl
            try:
                del self._train_dl
            except: pass
        except: pass

    def args_dl(self):
        config = {k:v for k,v in self.config.items() if k in self.KEYS_DL }
        return config

    @property
    def train_dl(self):
        try:
            return self._train_dl
        except:
            self._train_dl = DataLoader(self.train_ds, **self.args_dl(), )
            return self._train_dl
   
    @property
    def valid_dl(self):
        try:
            return self._valid_dl
        except:
            self._valid_dl = DataLoader(self.valid_ds, **self.args_dl(), )
            return self._valid_dl

    @property
    def valid_tensors(self):
        return self.valid_ds.tensors

    @property
    def train_tensors(self):
        return self.train_ds.tensors

    @property
    def train_X(self):
        return self.train_tensors[0]

    @property
    def train_y(self):
        return self.train_tensors[1]

    @property
    def valid_X(self):
        return self.valid_tensors[0]

    @property
    def valid_y(self):
        return self.valid_tensors[1]

    def train_numpy(self):
        return to_numpy(self.train_X), to_numpy(self.train_y)

    @property
    def valid_numpy(self):
        return to_numpy(self.valid_X), to_numpy(self.valid_y)

    def plot(self, **kwargs):
        return Plot(**kwargs)

    def plot_train(self, **kwargs):
        p = Plot(**kwargs)
        p.plot(self.train_X, self.train_y)
        return p

    def plot_valid(**kwargs):
        p = Plot(self, **kwargs)
        p.plot(self.valid_X, self.valid_y)
        return p

class Plot():
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.plt = plt.subplot(**kwargs)

    def plot( self, x=None, y=None, **kwargs):
        if x is not None:
            x = to_numpy(x).flatten()
            self.x = x
        y = to_numpy(y).flatten()
        self.plt.scatter(self.x, y, **kwargs)
