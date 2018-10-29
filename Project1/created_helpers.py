import numpy as np

def least_squares_cost(y, tx, w):
    '''
    :param y: np.array (m,1)
    :param tx: np.array (m,n)
    :param w: np.array (n,1)
    :return: float
    '''
    m = tx.shape[0]
    cost = np.sum(np.square(y - np.dot(tx, w))) / (2 * m)
    return cost


def sigmoid_fn(z):
    h = 1 / (1 + np.exp(-z))
    return h


def ridge_regression_cost(y, tx, w, lambda_):
    # assume w has no bias column

    m = tx.shape[0]
    a = np.sum(np.square(y - np.dot(tx, w)))
    b = lambda_ * np.sum(np.square(w))
    cost = (a + b) / (2 * m)
    return cost


# compute cost
def logistic_regression_cost(y, tx, w):
    z = tx.dot(w)
    h = sigmoid_fn(z)
    cost = y.T.dot(np.log(h)) + (1 - y).T.dot(np.log(1 - h))
    return np.squeeze(- cost)


# gradient of loss
def logistic_regression_gradient(y, tx, w):
    z = tx.dot(w)
    h = sigmoid_fn(z)
    grad = tx.T.dot(h - y)
    return grad


def reg_logistic_regression_cost_grad(y, tx, w, lambda_):
    loss = logistic_regression_cost(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
    gradient = logistic_regression_gradient(y, tx, w) + 2 * lambda_ * w
    return loss, gradient


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def compute_mse(y, tx, w):
    """compute the loss by mse."""
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse


def accuracy(y_pred, y_test):
    correct = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_test[i]:
            correct += 1
    return correct / len(y_test)


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree + 1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):  # taken from ex02 helper.py
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
