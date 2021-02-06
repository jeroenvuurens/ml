from sklearn.tree import *
from .classification import *

class decision_tree(classification):
    def __init__(self, data, max_depth=5, **kwargs):
        kwargs.update({'max_depth': max_depth})
        super().__init__(data, **kwargs)

    @property
    def model(self):
        try:
            return self._model
        except:
            self._model = DecisionTreeClassifier(max_depth=self.config.max_depth)
            return self._model

