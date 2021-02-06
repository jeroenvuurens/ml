from sklearn.svm import *
from sklearn.metrics import *
from .model import *
from .train_metrics import *
from .data import polynomials

class linear_svm(model):
    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)

    def predict(self, X):
        return self.model.predict(X)

    def step(self, X, y):
        self.model.fit(X, y)

    def parameters(self):
        return self.model.support_vectors_

    def plot_boundary(self, h=0.01):
        ax = plt.gca()
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        X = np.matrix(np.vstack([xx.ravel(), yy.ravel()])).T
        if self.data.degree > 1:
            X = polynomials(X, self.data.degree)
        boundary = self.predict(X)
        boundary = boundary.reshape(xx.shape)
        ax.contour(xx, yy, boundary)

    @property
    def model(self):
        try:
            return self._model
        except:
            self._model = SVC(gamma='scale')
            return self._model

