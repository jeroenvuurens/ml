from sklearn.svm import *
from .classification import *

class svm(classification):
    def parameters(self):
        return self.model.support_vectors_

    @property
    def model(self):
        try:
            return self._model
        except:
            self._model = SVC(kernel='linear', C=0.025)
            return self._model

