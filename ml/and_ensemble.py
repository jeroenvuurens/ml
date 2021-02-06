from sklearn.linear_model import *
from sklearn.metrics import accuracy_score, recall_score, precision_score
from .classification import *

class and_ensemble(classification):
    def __init__(self, data, *models):
        super().__init__(data)
        self.models = models

    def parameters(self):
        raise Exception('cannot show parameters for an and classifier')

    def model(self):
        return self

    def predict(self, X):
        return (np.sum([ m.predict(X) for m in self.models ], axis=0) == len(self.models)) * 1

    def accuracy(self, X, y):
        pred_y = self.predict(X)
        return accuracy_score(y, pred_y)

    def recall(self, X, y):
        pred_y = self.predict(X)
        return recall_score(y, pred_y)

    def precision(self, X, y):
        pred_y = self.predict(X)
        return precision_score(y, pred_y)

    def step(self, X, y):
        raise Exception('cannot train an and_classifier')


