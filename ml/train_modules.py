import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from sklearn.metrics import confusion_matrix


class training_module():
    def __init__(self, history):
        self.history = history
        self.trainer = history.trainer
        self.config = history.config
        
    def require_module(self, module):
        self.history.require_module(module)
    
    def requirements(self):
        return False
        
class store_y(training_module):
    def after_batch(self, epoch, X, y, y_pred, loss):
        if epoch['n'] == 0:
            epoch['y'] = y
            epoch['y_pred'] = y_pred
        else:
            epoch['y'] = np.vstack([epoch['y'], y])
            epoch['y_pred'] = np.vstack([epoch['y_pred'], y_pred])     

class store_contingencies(training_module):
    "stores a 2x2 contingency table"
    def after_batch(self, epoch, X, y, y_pred, loss):
        confusion_vector = np.round(y_pred) / y
        epoch['tp'] += np.sum(confusion_vector == 1)
        epoch['fp'] += np.sum(confusion_vector == float('inf'))
        epoch['tn'] += np.sum(np.isnan(confusion_vector))
        epoch['fn'] += np.sum(confusion_vector == 0)

class store_sse(training_module):
    def after_batch(self, epoch, X, y, y_pred, loss):
        epoch['sse'] += np.sum((y_pred - y)**2)

class store_ssy(training_module):
    def after_batch(self, epoch, X, y, y_pred, loss):
        mean_y = np.sum(y) / len(y)
        epoch['ssy'] += np.sum((y - mean_y)**2)

class store_sae(training_module):
    def after_batch(self, epoch, X, y, y_pred, loss):
        epoch['sae'] += np.sum(np.abs(y_pred - y))
    
