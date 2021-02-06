def order_x(X):
        return X[X[:,0].argsort(axis=0), :]

