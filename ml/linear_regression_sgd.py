import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
from sklearn.linear_model import *
from .regression import *
from .train_metrics import *

class linear_regression_sgd(regression):
    def __init__(self, data, metrics=[loss], lr=0.01, adaptive='invscaling', **kwargs):
        kwargs['metrics'] = metrics
        kwargs['lr'] = lr
        kwargs['adaptive'] = adaptive
        super().__init__(data, **kwargs)

    def predict(self, X):
        return self.model.predict(X)

    def step(self, X, y):
        self.model.partial_fit(X, y)

    @property
    def model(self):
        try:
            return self._model
        except:
            self._model = SGDRegressor(eta0=self.config.lr, learning_rate=self.config.adaptive)
            return self._model

