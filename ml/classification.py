from sklearn.neighbors import *
from sklearn.metrics import *
from .model import *
from .train_metrics import *
from .data import polynomials

def nll(X, y):
    return log_loss(X, y) * len(X)

class classification(model):
    def __init__(self, data, loss=nll, metrics=[loss, acc], **kwargs):
        super().__init__(data, loss=loss, metrics=metrics, **kwargs)

    def predict(self, X):
        return self.model.predict(X)

    def predict_p(self, X):
        return self.model.predict(X)

    def step(self, X, y):
        self.model.fit(X, y)

    def parameters(self):
        return np.hstack([np.array([self.model.intercept_]), self.model.coef_])

    def plot_boundary(self):
        ax = plt.gca()
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        stepx = (x_max - x_min) / 600
        stepy = (y_max - y_min) / 600
        xx, yy = np.meshgrid(np.arange(x_min, x_max, stepx),
                             np.arange(y_min, y_max, stepy))
        X = np.matrix(np.vstack([xx.ravel(), yy.ravel()])).T
        if self.data.degree > 1:
            X = polynomials(X, self.data.degree)
        boundary = self.predict(X)
        boundary = boundary.reshape(xx.shape)
        ax.contour(xx, yy, boundary, levels=[0.5])

