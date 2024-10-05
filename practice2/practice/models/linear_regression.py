import numpy as np


class LinearRegression:
    def __init__(self, regularization_type: str="l2", alpha: float = 0.01, max_iter: int=1500, tol: float = 1e-7):

        assert regularization_type in ['l1', 'l2', None], f"regularization_type {regularization_type} is not supported"

        # bias is inside
        self._weights = None

        # b + w1x1 + w2x2 + ...

        # [b, w1, w2, ...] @ [1, x1, x2,...]
        #
        # [1, x1, x2,...]
        # [1, x1, x2,...]
        # [1, x1, x2,...]
        # [1, x1, x2,...]
        #  ....


        self._reg_type = regularization_type
        self._alpha = alpha
        self._max_iter = max_iter
        self._tol = tol



    def _least_squares(self, X: np.ndarray, y: np.ndarray):

        assert self._reg_type in ["l2", None], f"{self._reg_type} is not supported in least squares"

        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        I = np.eye(X_b.shape[1])
        I[0, 0] = 0

        if self._reg_type is None:
            I *= 0.0

        self._weights = np.linalg.inv(X_b.T @ X_b + self._alpha * I) @ X_b.T @ y


    def _gradient_descent(self, X: np.ndarray, y: np.ndarray, learning_rate: float):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        self._weights = np.random.rand(X_b.shape[1])
        n = len(y)

        for _ in range(self._max_iter):
            pred = X_b @ self._weights
            error = pred - y
            gradient = 1 / n * X_b.T @ error

            if self._reg_type == "l1":
                gradient[1:] += (self._alpha / n) * np.sign(self._weights[1:])
            elif self._reg_type == "l2":
                gradient[1:] += (self._alpha / n) * self._weights[1:]

            prev_weights = self._weights.copy()
            self._weights -= learning_rate * gradient

            if np.linalg.norm(self._weights - prev_weights, ord=1) < self._tol:
                break


    def fit(self, X: np.ndarray, y: np.ndarray, method: str="least_squares", learning_rate: float=0.01):
        """
        Fits linear regression with a aseleceted method.

        Args:
            X (np.ndarray): features (n_samples, n_features)
            y (np.ndarray): target vector (n_samples,)
            method (str, optional): "least_squares" | "gradient_descent". Defaults to "least_squares".
        """

        if method == "least_squares":
            self._least_squares(X, y)
        elif method == "gradient_descent":
            self._gradient_descent(X, y, learning_rate=learning_rate)
        else:
            raise ValueError('Invalid method')


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions using fitted model.

        Args:
            X (np.ndarray): Features (n_samples, n_features)

        Returns:
            np.ndarray: predictions
        """

        if self._weights is None:
            raise ValueError('Model not fitted')

        # stack, concatenate, hstack, vstack
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self._weights

