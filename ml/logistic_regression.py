from sklearn.linear_model import *
from .classification import *

class logistic_regression(classification):
    def __init__(self, data, lrargs={}, **kwargs):
        super().__init__(data, **kwargs)
        self.lrargs = lrargs

    def parameters(self):
        return (self.model.intercept_, self.model.coef_)

    @property
    def model(self):
        try:
            return self._model
        except:
            self._model = LogisticRegression(**self.lrargs)
            return self._model

