import numpy as np


def compute_loss(y, tx, w):
    """Compute the mean squared error loss, with the 1/2 factor: L(w) = 1/(2N) * sum((y - tx @ w)**2).

    Args:
        y: shape=(N,). Target values.
        tx: shape=(N, D). Feature matrix.
        w: shape=(D,). The vector of model parameters.

    Returns:
        loss: scalar, the MSE loss at w.
    """
    N = y.shape[0]
    y_pred = tx @ w
    e = y - y_pred

    loss = (1 / (2 * N)) * np.sum(e**2)

    return loss


# Linear regression using gradient descent
def compute_gradient(y, tx, w):
    """Compute the gradient of the MSE loss at w: grad = -1/N * tx.T @ (y - tx @ w).

    Args:
        y: shape=(N,). Target values.
        tx: shape=(N, D). Feature matrix.
        w: shape=(D,). The vector of model parameters.

    Returns:
        gradient: shape=(D,) (same shape as w), the gradient of the MSE loss at w.
    """
    N = y.shape[0]
    y_pred = tx @ w
    e = y - y_pred

    gradient = -1 / N * (tx.T @ e)

    return gradient


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Fit a linear regression model by full-batch gradient descent on the MSE loss.

    Args:
        y: shape=(N,). Target values.
        tx: shape=(N, D). Feature matrix.
        initial_w: shape=(D,). Initial guess for the model parameters.
        max_iters: int, total number of gradient descent steps.
        gamma: float, step size.

    Returns:
        w: shape=(D,), model parameters after the last iteration.
        loss: scalar, MSE loss at the final w.
    """
    # Define parameters to store w and loss
    w = initial_w

    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        w = w - gamma * gradient  # updates w value

    loss = compute_loss(y, tx, w)
    return w, loss


# Linear regression using stochastic gradient descent
def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient of the MSE loss at w from a subset of samples.

    In mean_squared_error_sgd this is called with a single sample (batch size 1).

    Args:
        y: shape=(B,). Target values of the sampled batch (B = 1 here).
        tx: shape=(B, D). Feature rows of the sampled batch.
        w: shape=(D,). The vector of model parameters.

    Returns:
        gradient: shape=(D,) (same shape as w), the stochastic gradient of the MSE loss at w.
    """
    N = y.shape[0]
    y_pred = tx @ w
    e = y - y_pred

    gradient = -1 / N * (tx.T @ e)
    return gradient


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Fit a linear regression model by stochastic gradient descent on the MSE loss.

    Each step uses one randomly chosen sample (mini-batch size 1).

    Args:
        y: shape=(N,). Target values.
        tx: shape=(N, D). Feature matrix.
        initial_w: shape=(D,). Initial guess for the model parameters.
        max_iters: int, total number of SGD steps.
        gamma: float, step size.

    Returns:
        w: shape=(D,), model parameters after the last iteration.
        loss: scalar, MSE loss of the final w, computed on the full dataset.
    """

    # Define parameters
    w = initial_w
    N = y.shape[0]

    for n_iter in range(max_iters):
        # pick one random example
        n = np.random.randint(N)
        y_n = y[n : n + 1]
        tx_n = tx[n : n + 1]

        gradient = compute_stoch_gradient(y_n, tx_n, w)
        w = w - gamma * gradient  # updates w value

    loss = compute_loss(y, tx, w)

    return w, loss


# Least squares regression using normal equations
def least_squares(y, tx):
    """Compute the least squares solution by solving the normal equations (tx.T @ tx) w = tx.T @ y.

    Args:
        y: shape=(N,). Target values.
        tx: shape=(N, D). Feature matrix.

    Returns:
        w: shape=(D,), optimal weights.
        loss: scalar, MSE loss (with 1/2 factor) at w.
    """

    N = y.shape[0]
    w = np.linalg.inv(tx.T @ tx) @ tx.T @ y

    loss = compute_loss(y, tx, w)

    return w, loss


# Ridge regression using normal equation
def ridge_regression(y, tx, lambda_):
    """Compute the ridge regression solution using the normal equations.

    Minimizes 1/(2N) * ||y - tx @ w||^2 + lambda_ * ||w||^2, whose solution is
    w = (tx.T @ tx + 2N * lambda_ * I)^(-1) @ tx.T @ y.

    Args:
        y: shape=(N,). Target values.
        tx: shape=(N, D). Feature matrix.
        lambda_: float, regularization strength.

    Returns:
        w: shape=(D,), optimal weights.
        loss: scalar, MSE loss (with 1/2 factor) at w, WITHOUT the penalty term.
    """
    N, D = y.shape[0], tx.shape[1]
    lambda_prime = 2 * N * lambda_

    w_ridge = np.linalg.inv(tx.T @ tx + lambda_prime * np.identity(D)) @ tx.T @ y
    loss = compute_loss(y, tx, w_ridge)

    return w_ridge, loss


# Logistic regression using gradient descent (y ∈ {0,1})
def sigmoid(t):
    """Apply the sigmoid function 1 / (1 + exp(-t)) elementwise.

    Args:
        t: scalar or numpy array.

    Returns:
        scalar or numpy array of the same shape as t, with values in (0, 1).
    """
    return 1 / (1 + np.exp(-t))


# small function to find logistic loss more easily
def logistic_loss(y, tx, w):
    """Mean negative log-likelihood for logistic regression.

    Args:
        y: shape=(N,), labels in {0, 1}.
        tx: shape=(N, D), feature matrix.
        w: shape=(D,), model weights.

    Returns:
        loss: scalar, mean negative log-likelihood (no penalty term).
    """

    t = tx @ w
    log_loss = np.mean(
        np.logaddexp(0, t) - y * t
    )  # same as eqn in other functions, just written so no overflow when t is large

    return log_loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Fit a logistic regression classifier by full-batch gradient descent.

    Models P(y_n = 1 | x_n) = sigmoid(x_n @ w) and minimizes the mean negative
    log-likelihood: 1/N * sum(log(1 + exp(tx @ w)) - y * (tx @ w)).

    Args:
        y: shape=(N,). Labels in {0, 1}.
        tx: shape=(N, D). Feature matrix.
        initial_w: shape=(D,). Initial guess for the model parameters.
        max_iters: int, total number of gradient descent steps.
        gamma: float, step size.

    Returns:
        w: shape=(D,), model parameters after the last iteration.
        loss: scalar, mean negative log-likelihood at the final w.
    """
    # Define parameters
    w = initial_w
    N = y.shape[0]

    for n_iter in range(max_iters):
        gradient = (1 / N) * tx.T @ (sigmoid(tx @ w) - y)
        w = w - gamma * gradient

    loss = (1 / N) * np.sum(-y * (tx @ w) + np.log(1 + np.exp(tx @ w)))

    return w, loss


# Regularized logistic regression using gradient descent (y ∈ {0,1}, with regularization term λ∥w∥2)
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Fit an L2-regularized logistic regression classifier by full-batch gradient descent.

    Minimizes the mean negative log-likelihood plus lambda_ * ||w||^2, so each
    gradient step includes an extra 2 * lambda_ * w term. All weights,
    including the bias (if tx has a column of ones), are penalized.

    Args:
        y: shape=(N,). Labels in {0, 1}.
        tx: shape=(N, D). Feature matrix.
        lambda_: float, regularization strength.
        initial_w: shape=(D,). Initial guess for the model parameters.
        max_iters: int, total number of gradient descent steps.
        gamma: float, step size.

    Returns:
        w: shape=(D,), model parameters after the last iteration.
        loss: scalar, mean negative log-likelihood at the final w, WITHOUT the penalty term.
    """
    # Define parameters
    w = initial_w
    N = y.shape[0]

    for n_iter in range(max_iters):
        gradient = (1 / N) * tx.T @ (sigmoid(tx @ w) - y) + 2 * lambda_ * w
        w = w - gamma * gradient

    loss = (1 / N) * np.sum(-y * (tx @ w) + np.log(1 + np.exp(tx @ w)))

    return w, loss
