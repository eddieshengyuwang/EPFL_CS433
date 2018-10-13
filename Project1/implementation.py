import numpy as np
from proj1_helpers import load_csv_data, predict_labels, create_csv_submission
<<<<<<< HEAD
from created_helpers import batch_iter, least_squares_cost


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    '''
    :param y: labels -> m x 1 np.array
    :param tx: X -> m x n-1 np.array
    :param initial_w: weights -> n x 1 np.array
    :param max_iters: int
    :param gamma: step size -> float
    :return: (w, loss) -> (np.array, int)
    '''

    m = y.shape[0]
    for i in max_iters:
        error = np.dot(tx, initial_w) - y  # (m,1)
        gradient = np.dot(tx.T, error)  # (n,m)x(m,1) -> (n,1)
        initial_w = initial_w - (gamma / m) * gradient

    cost = least_squares_cost(y, tx, initial_w)
    return (initial_w, cost)


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):

    m = y.shape[0]
    for i in max_iters:
        for minibatch_y, minibatch_tx in batch_iter(y, tx, 1): # O(1) runtime
            error = np.dot(minibatch_tx, initial_w) - minibatch_y  # (1,1)
            gradient = np.dot(minibatch_tx.T, error)  # (n,1)x(1,1) -> (n,1)
            initial_w = initial_w - (gamma / m) * gradient

    cost = least_squares_cost(y, tx, initial_w)
    return (initial_w, cost)

def least_squares(y, tx):


=======

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    pass

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    pass

def least_squares(y, tx):
>>>>>>> c148e344de3fa9ef24c892699a4f061eab4af950
    pass

def ridge_regression(y, tx, lambda_):
    pass

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    pass

def reg_logistic_regression(y, tx, lambda_, inital_w, max_iters, gamma):
<<<<<<< HEAD
    pass
=======
    pass

>>>>>>> c148e344de3fa9ef24c892699a4f061eab4af950
