from sklearn.linear_model import *
from .regression import *
from .train_metrics import *

class linear_regression_ne(regression):
    """Solve Linear Regression problems using the Normal Equation"""
    def __init__(self, data, **kwargs):
        kwargs.setdefault('metrics', [ loss ])
        super().__init__(data, **kwargs)

    def predict(self, X):
        return self.model.predict(X)

    def step(self, X, y):
        self.model.fit(X, y)

    @property
    def model(self):
        try:
            return self._model
        except:
            self._model = LinearRegression()
            return self._model

