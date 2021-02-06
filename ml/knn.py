from sklearn.neighbors import *
from .classification import *

class knn(classification):
    def __init__(self, data, k=1, **kwargs):
        kwargs.update({'k':k})
        super().__init__(data, **kwargs)

    @property
    def model(self):
        try:
            return self._model
        except:
            self._model = KNeighborsClassifier(self.config.k)
            return self._model

