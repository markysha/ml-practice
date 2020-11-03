from __future__ import annotations

import numpy as np
import copy
#test

s0_default: float = 1
p_default: float = 0.5

batch_size_default: int = 1

alpha_default: float = 0.1
eps_default: float = 1e-8

mu_default = 1e-2

tolerance_default: float = 1e-3
max_iter_default: int = 1000


class BaseDescent:
    """
    A base class and examples for all functions
    """

    def __init__(self):
        self.w = None

    def step(self, X: np.ndarray, y: np.ndarray, iteration: int) -> np.ndarray:
        """
        Descent step
        :param iteration: iteration number
        :param X: objects' features
        :param y: objects' targets
        :return: difference between weights
        """
        return self.update_weights(self.calc_gradient(X, y), iteration)

    def update_weights(self, gradient: np.ndarray, iteration: int) -> np.ndarray:
        """
        Example for update_weights function
        :param iteration: iteration number
        :param gradient: gradient
        :return: weight difference: np.ndarray
        """
        pass

    def calc_gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Example for calc_gradient function
        :param X: objects' features
        :param y: objects' targets
        :return: gradient: np.ndarray
        """
        pass


class GradientDescent(BaseDescent):
    """
    Full gradient descent class
    """

    def __init__(self, w0: np.ndarray, lambda_: float, s0: float = s0_default, p: float = p_default):
        """
        :param w0: weight initialization
        :param lambda_: learning rate parameter (float)
        :param s0: learning rate parameter (float)
        :param p: learning rate parameter (float)
        """
        super().__init__()
        self.eta = lambda k: lambda_ * (s0 / (s0 + k)) ** p
        self.w = np.copy(w0)

    def update_weights(self, gradient: np.ndarray, iteration: int) -> np.ndarray:
        """
        Changing weights with respect to gradient
        :param iteration: iteration number
        :param gradient: gradient
        :return: weight difference: np.ndarray
        """
        delta = gradient * self.eta(iteration)
        self.w = self.w - delta
        return delta 

    def calc_gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Getting objects, calculating gradient at point w
        :param X: objects' features
        :param y: objects' targets
        :return: gradient: np.ndarray
        """
        # TODO: implement calculating gradient function
        # raise NotImplementedError('GradientDescent calc_gradient function not implemented')
#         return (X * np.array([(X.dot(self.w) - y)]).T).mean(axis=0)
        return (X.T.dot(X.dot(self.w) - y)) * 2.0 / X.shape[0]


class StochasticDescent(BaseDescent):
    """
    Stochastic gradient descent class
    """

    def __init__(self, w0: np.ndarray, lambda_: float, s0: float = s0_default, p: float = p_default,
                 batch_size: int = batch_size_default):
        """
        :param w0: weight initialization
        :param lambda_: learning rate parameter (float)
        :param s0: learning rate parameter (float)
        :param p: learning rate parameter (float)
        :param batch_size: batch size (int)
        """
        super().__init__()
        self.eta = lambda k: lambda_ * (s0 / (s0 + k)) ** p
        self.batch_size = batch_size
        self.w = np.copy(w0)

    def update_weights(self, gradient: np.ndarray, iteration: int) -> np.ndarray:
        """
        Changing weights with respect to gradient
        :param iteration: iteration number
        :param gradient: gradient estimate
        :return: weight difference: np.ndarray
        """
        delta = gradient * self.eta(iteration)
        self.w = self.w - delta
        return delta 

    def calc_gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Getting objects, calculating gradient at point w
        :param X: objects' features
        :param y: objects' targets
        :return: gradient: np.ndarray
        """
        indexes = np.random.randint(X.shape[0], size=self.batch_size)
        X_batch = X[indexes, :]
        y_batch = y[indexes]
        return (X_batch.T.dot(X_batch.dot(self.w) - y_batch)) * 2.0 / X_batch.shape[0]


class MomentumDescent(BaseDescent):
    """
    Momentum gradient descent class
    """

    def __init__(self, w0: np.ndarray, lambda_: float, alpha: float = alpha_default, s0: float = s0_default,
                 p: float = p_default):
        """
        :param w0: weight initialization
        :param lambda_: learning rate parameter (float)
        :param alpha: momentum coefficient
        :param s0: learning rate parameter (float)
        :param p: learning rate parameter (float)
        """
        super().__init__()
        self.eta = lambda k: lambda_ * (s0 / (s0 + k)) ** p
        self.alpha = alpha
        self.w = np.copy(w0)
        self.h = 0

    def update_weights(self, gradient: np.ndarray, iteration: int) -> np.ndarray:
        """
        Changing weights with respect to gradient
        :param iteration: iteration number
        :param gradient: gradient estimate
        :return: weight difference: np.ndarray
        """
        self.h = self.alpha * self.h + gradient * self.eta(iteration)
        self.w = self.w - self.h
        return self.h 

    def calc_gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Getting objects, calculating gradient at point w
        :param X: objects' features
        :param y: objects' targets
        :return: gradient: np.ndarray
        """
        return (X.T.dot(X.dot(self.w) - y)) * 2.0 / X.shape[0]


class Adagrad(BaseDescent):
    """
    Adaptive gradient algorithm class
    """

    def __init__(self, w0: np.ndarray, lambda_: float, eps: float = eps_default, s0: float = s0_default,
                 p: float = p_default):
        """
        :param w0: weight initialization
        :param lambda_: learning rate parameter (float)
        :param eps: smoothing term (float)
        :param s0: learning rate parameter (float)
        :param p: learning rate parameter (float)
        """
        super().__init__()
        self.eta = lambda k: lambda_ * (s0 / (s0 + k)) ** p
        self.eps = eps
        self.w = np.copy(w0)
        self.g = 0

    def update_weights(self, gradient: np.ndarray, iteration: int) -> np.ndarray:
        """
        Changing weights with respect to gradient
        :param iteration: iteration number
        :param gradient: gradient estimate
        :return: weight difference: np.ndarray
        """
        # TODO: implement updating weights function
        self.g = self.g + gradient ** 2
        delta = self.eta(iteration) / ((self.eps + self.g) ** 0.5) * gradient
        self.w = self.w - delta
        return delta 

    def calc_gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Getting objects, calculating gradient at point w
        :param X: objects' features
        :param y: objects' targets
        :return: gradient: np.ndarray
        """
        # TODO: implement calculating gradient function
        return (X.T.dot(X.dot(self.w) - y)) * 2.0 / X.shape[0]


class GradientDescentReg(GradientDescent):
    """
    Full gradient descent with regularization class
    """

    def __init__(self, w0: np.ndarray, lambda_: float, mu: float = mu_default, s0: float = s0_default,
                 p: float = p_default):
        """
        :param mu: l2 coefficient
        """
        super().__init__(w0=w0, lambda_=lambda_, s0=s0, p=p)
        self.mu = mu

    def update_weights(self, gradient: np.ndarray, iteration: int) -> np.ndarray:
        return super().update_weights(gradient, iteration)

    def calc_gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        l2 = self.w
        return super().calc_gradient(X, y) + l2 * self.mu


class StochasticDescentReg(StochasticDescent):
    """
    Stochastic gradient descent with regularization class
    """

    def __init__(self, w0: np.ndarray, lambda_: float, mu: float = mu_default, s0: float = s0_default,
                 p: float = p_default, batch_size: int = batch_size_default):
        """
        :param mu: l2 coefficient
        """
        super().__init__(w0=w0, lambda_=lambda_, s0=s0, p=p, batch_size=batch_size)
        self.mu = mu

    def update_weights(self, gradient: np.ndarray, iteration: int) -> np.ndarray:
        return super().update_weights(gradient, iteration)

    def calc_gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        l2 = self.w
        return super().calc_gradient(X, y) + l2 * self.mu


class MomentumDescentReg(MomentumDescent):
    """
    Momentum gradient descent with regularization class
    """

    def __init__(self, w0: np.ndarray, lambda_: float, alpha: float = alpha_default, mu: float = mu_default,
                 s0: float = s0_default, p: float = p_default):
        """
        :param mu: l2 coefficient
        """
        super().__init__(w0=w0, lambda_=lambda_, alpha=alpha, s0=s0, p=p)
        self.mu = mu

    def update_weights(self, gradient: np.ndarray, iteration: int) -> np.ndarray:
        return super().update_weights(gradient, iteration)

    def calc_gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        l2 = self.w
        return super().calc_gradient(X, y) + l2 * self.mu


class AdagradReg(Adagrad):
    """
    Adaptive gradient algorithm with regularization class
    """

    def __init__(self, w0: np.ndarray, lambda_: float, eps: float = eps_default, mu: float = mu_default,
                 s0: float = s0_default, p: float = p_default):
        """
        :param mu: l2 coefficient
        """
        super().__init__(w0=w0, lambda_=lambda_, eps=eps, s0=s0, p=p)
        self.mu = mu

    def update_weights(self, gradient: np.ndarray, iteration: int) -> np.ndarray:
        return super().update_weights(gradient, iteration)

    def calc_gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        l2 = self.w
        return super().calc_gradient(X, y) + l2 * self.mu


class LinearRegression:
    """
    Linear regression class
    """

    def __init__(self, descent, tolerance: float = tolerance_default, max_iter: int = max_iter_default):
        """
        :param descent: Descent class
        :param tolerance: float stopping criterion for square of euclidean norm of weight difference
        :param max_iter: int stopping criterion for iterations
        """
        self.descent = descent
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.loss_history = []
        self.w = self.descent.w

    def fit(self, X: np.ndarray, y: np.ndarray) -> LinearRegression:
        """
        Getting objects, fitting descent weights
        :param X: objects' features
        :param y: objects' target
        :return: self
        """
        self.descent.w = np.ones(X.shape[1])
        self.w = self.descent.w.copy()
        i = 0
        
        self.descent.h = 0
        self.descent.g = 0
        self.descent.d = 0
        
        while i < self.max_iter:
            self.calc_loss(X, y)
            delta = self.descent.step(X, y, i)
            self.w = self.w - delta
            if np.sum(delta ** 2) < self.tolerance:
                break
            i = i + 1
        
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Getting objects, predicting targets
        :param X: objects' features
        :return: predicted targets
        """
        return np.matmul(X, np.ones(X.shape[1]) * self.w)

    def calc_loss(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Getting objects, calculating loss
        :param X: objects' features
        :param y: objects' target
        """
        self.loss_history.append(np.mean((self.predict(X) - y) ** 2))


###########################################################
####################### BONUS TASK ########################
###########################################################


class StochasticAverageGradient(BaseDescent):
    """
    Stochastic average gradient class (BONUS TASK)
    """

    def __init__(self, w0: np.ndarray, lambda_: float, x_shape: int, s0: float = s0_default, p: float = p_default):
        """
        :param w0: weight initialization
        :param lambda_: learning rate parameter (float)
        :param s0: learning rate parameter (float)
        :param p: learning rate parameter (float)
        """
        super().__init__()
        self.eta = lambda k: lambda_ * (s0 / (s0 + k)) ** p
        self.lambda_ = lambda_
        self.w = np.copy(w0)
        # gradient for every row of data
        self.v = np.array([[np.nan]]) #np.zeros((x_shape, w0.shape[0]))
        self.d = 0

    def update_weights(self, gradient: np.ndarray, iteration: int) -> np.ndarray:
        """
        Changing weights with respect to gradient
        :param iteration: iteration number
        :param gradient: gradient
        :return: weight difference: np.ndarray
        """
        delta = gradient * self.lambda_
        self.w = self.w - delta
        return delta 

    def calc_gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Getting objects, calculating gradient at point w
        :param X: objects' features
        :param y: objects' targets
        :return: gradient: np.ndarray
        """
        if np.isnan(self.v[0, 0]):
            # if it is the first iteration
            self.v = np.zeros(X.shape) #X * np.tile((X.dot(self.w) - y), (X.shape[1], 1)).T 
            self.d = np.sum(self.v, axis=0)
        
        index = np.random.randint(X.shape[0], size=100)
        v_index = np.tile((X[index, :].dot(self.w) - y[index]), (X.shape[1], 1)).T * X[index, :]
        
        self.d -= np.sum(self.v[index, :], axis=0)
        self.v[index, :] = v_index
        self.d += np.sum(self.v[index, :], axis=0)
        return self.d * 2.0 / X.shape[0]

###########################################################
####################### BONUS TASK ########################
###########################################################
