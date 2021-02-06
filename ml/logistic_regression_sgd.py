from sklearn.linear_model import *
from .classification import *
from scipy.special import expit as logit

def J(X, y, 𝜃):
    return - 1 / len(X) * (y.T @ np.log(logit(X @ 𝜃)) + (1 - y.T) @ np.log(1- logit(X @ 𝜃)))

def h(X, 𝜃):
    return logit(X @ 𝜃)

class logistic_regression_sgd(classification):
    def __init__(self, data, lr=0.03, **kwargs):
        kwargs['lr'] = lr
        data.bias = True
        data.column_y = True
        super().__init__(data, **kwargs)

    def parameters(self):
        return self.model
    
    def step(self, X, y):
        self.m = X.shape[1]            # the number of features
        self.model -= (self.config.lr / self.m) * X.T @ ( h(X, self.model) - y )

    def predict(self, X):
        return h(X, self.model) >= 0.5

    @property
    def model(self):
        try:
            return self._model
        except:
            self._model = np.zeros((self.m, 1))
            return self._model

    @model.setter
    def model(self, value):
        self._model = value
