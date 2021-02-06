from .train_modules import *
from sklearn.metrics import accuracy_score
from math import *

def name_metrics(metrics):
    for i, m in enumerate(metrics):
        try:
            m, m.__name__ = m.value, m.__name__
        except:
            m.__name__ = f'm_{i}'
        yield m

class train_metrics():
    def __init__(self, history=None):
        self.history = history
        self.config = history.config
        
    def requirements(self):
        pass
    
    @property
    def __name__(self):
        return self.__name()

    def __name(self):
        return self.__class__.__name__
  
    @staticmethod
    def value(epoch):
        pass
        
class acc(train_metrics):
    def requirements(self):
        self.history.require_module(store_contingencies)
    
    @staticmethod
    def value(epoch):
        return (epoch['tp'] + epoch['tn']) / epoch['n']

class acc_mc(train_metrics):
    def requirements(self):
        self.history.require_module(store_confusion)
    
    @staticmethod
    def value(epoch):
        return np.diag(epoch['cm']).sum() / epoch['n']

class recall(train_metrics):
    def requirements(self):
        self.history.require_module(store_contingencies)

    @staticmethod
    def value(epoch):
        return epoch['tp'] / (epoch['tp'] + epoch['fn']) 

class precision(train_metrics):
    def requirements(self):
        self.history.require_module(store_contingencies)

    @staticmethod
    def value(epoch):
        return epoch['tp'] / (epoch['tp'] + epoch['fp']) 

class f1(train_metrics):
    def requirements(self):
        self.history.require_module(store_contingencies)

    @staticmethod
    def value(epoch):
        r = recall.value(epoch)
        p = precision.value(epoch)
        return 2 * r * p / (r + p)

class mse(train_metrics):
    def requirements(self):
        self.history.require_module(store_sse)

    @staticmethod
    def value(epoch):
        return epoch['sse'] / epoch['n']

class mae(train_metrics):
    def requirements(self):
        self.history.require_module(store_sae)

    @staticmethod
    def value(epoch):
        return epoch['sae'] / epoch['n']

class rmse(train_metrics):
    def requirements(self):
        self.history.require_module(store_sse)

    @staticmethod
    def value(epoch):
        return sqrt(epoch['sse'] / epoch['n'])

class r2(train_metrics):
    def requirements(self):
        self.history.require_module(store_sse)
        self.history.require_module(store_ssy)

    @staticmethod
    def value(epoch):
        return 1- epoch['sse'] / epoch['ssy']

class loss(train_metrics):
    @staticmethod
    def value(epoch):
        return epoch['loss']    
    
