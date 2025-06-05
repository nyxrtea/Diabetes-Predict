import numpy as np

def rbf_kernel(x1, x2, gamma):
    return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)

class SVM_RBF:
    def __init__(self, C=1.0, gamma=0.5, lr=0.001, n_iters=100):
        self.C = C
        self.gamma = gamma
        self.lr = lr
        self.n_iters = n_iters

    def fit(self, X, y):
        self.X = X
        self.y = np.where(y <= 0, -1, 1)
        n_samples = X.shape[0]
        self.alpha = np.zeros(n_samples)
        self.b = 0

        self.K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                self.K[i, j] = rbf_kernel(X[i], X[j], self.gamma)

        for _ in range(self.n_iters):
            for i in range(n_samples):
                condition = np.sum(self.alpha * self.y * self.K[:, i]) + self.b
                if self.y[i] * condition < 1:
                    self.alpha[i] += self.lr * (1 - self.y[i] * condition)
                else:
                    self.alpha[i] -= self.lr * self.alpha[i] * self.C

    def project(self, X):
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            s = 0
            for alpha_i, y_i, x_i in zip(self.alpha, self.y, self.X):
                if alpha_i > 1e-5:
                    s += alpha_i * y_i * rbf_kernel(X[i], x_i, self.gamma)
            y_pred[i] = s + self.b
        return y_pred

    def predict(self, X):
        return np.where(self.project(X) >= 0, 1, 0)
