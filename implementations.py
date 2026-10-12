import numpy as np

def compute_loss(y, tx, w):
    """Calculate the mean squared error

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    N = y.shape[0]
    y_pred = tx @ w
    e = y - y_pred

    loss = (1 / N) * np.sum(e**2)

    return loss

# Linear regression using gradient descent
def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    N = y.shape[0]
    y_pred = tx @ w
    e = y - y_pred

    gradient = -1/N * (tx.T @ e)

    return gradient

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w) 
        w = w - gamma * gradient # updates w value

        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return ws, losses

# Linear regression using stochastic gradient descen
def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """
    N = y.shape[0]
    y_pred = tx @ w
    e = y - y_pred

    gradient = -1/N * (tx.T @ e)
    return gradient

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """

    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    N = y.shape[0]

    for n_iter in range(max_iters):
        # pick one random example
        n = np.random.randint(N)
        y_n = y[n:n+1]
        tx_n = tx[n:n+1]

        gradient = compute_stoch_gradient(y_n, tx_n, w)
        w = w - gamma * gradient #updates w value

        loss = compute_loss(y, tx, w)

        # store w and loss
        ws.append(w)
        losses.append(loss)

        print("SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    
    return ws, losses

# Least squares regression using normal equations
def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.
    """

    N = y.shape[0]
    w = np.linalg.inv(tx.T @ tx) @ tx.T @ y

    loss = compute_loss(y, tx, w)

    return w, loss

# Ridge regression using normal equation
def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:

        w: optimal weights, numpy array of shape(D,), D is the number of features.
    """
    N, D = y.shape[0], tx.shape[1]
    lambda_prime = 2 * N * lambda_

    w_ridge = np.linalg.inv(tx.T @ tx + lambda_prime * np.identity(D)) @ tx.T @ y
    loss = compute_loss(y, tx, w_ridge)
    
    return w_ridge, loss

# Logistic regression using gradient descent (y ∈ {0,1})

# Regularized logistic regression using gradient descent (y ∈ {0,1}, with regularization term λ∥w∥2)