from sklearn.metrics import *
from .model import *

def squared_error(y_pred, y):
    return mean_squared_error(y_pred, y) * len(y)

class regression(model):
    def __init__(self, data, loss=squared_error, metrics=[ loss ], **kwargs):
        super().__init__(data, loss=loss, metrics=metrics, **kwargs)

    def plot_train_line(self, marker='-', interpolate=2, **kwargs):
        X = self.data.train_X_interpolated(interpolate)
        self.plot_X(X, marker=marker, **kwargs)
        
    def plot_valid_line(self, marker='-', interpolate=0, **kwargs):
        X = self.data.valid_X_interpolated(interpolate)
        self.plot_X(X, marker=marker, **kwargs)

    def plot_line(self, marker='-', interpolate=2, **kwargs):
        X = self.data.X_interpolated(interpolate)
        self.plot_X(X, marker=marker, **kwargs)

    def parameters(self):
        return np.hstack([np.array([self.model.intercept_]), self.model.coef_])

