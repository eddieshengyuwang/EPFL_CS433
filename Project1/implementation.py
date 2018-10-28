'''
NOTE that in our implementation, we use m to represent # rows
and n to represent # columns
'''

import numpy as np
from created_helpers import *

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
    for i in range(max_iters):
        error = np.dot(tx, initial_w) - y  # (m,1)
        gradient = np.dot(tx.T, error)  # (n,m)x(m,1) -> (n,1)
        initial_w = initial_w - (gamma / m) * gradient

    cost = least_squares_cost(y, tx, initial_w) # imported from created_helpers
    return (initial_w, cost)


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):

    m = y.shape[0]
    for i in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, 1): # O(1) runtime
            error = np.dot(minibatch_tx, initial_w) - minibatch_y  # (1,1)
            gradient = np.dot(minibatch_tx.T, error)  # (n,1)x(1,1) -> (n,1)
            initial_w = initial_w - (gamma / m) * gradient

    cost = least_squares_cost(y, tx, initial_w) # imported from created_helpers

    return (initial_w, cost)

def least_squares(y, tx):
    # equation is (X.T*X)^(-1) * (X.T*y)
    first_term = tx.T.dot(tx)
    second_term = tx.T.dot(y)
    optimal_weight = np.linalg.solve(first_term, second_term)
    cost = least_squares_cost(y, tx, optimal_weight) # imported from created_helpers
    return (optimal_weight, cost) # (n,n)x(n,1) -> (n,1)

    
def ridge_regression(y, tx, lamb):
    """implement ridge regression."""
    print("what")
    aI = lamb * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    optimal_weight = np.linalg.solve(a, b)
    print("bye")
    cost = ridge_regression_cost(y, tx, optimal_weight, lamb) # imported from created_helpers
    print("hi")
    return (optimal_weight, cost)
        
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    m = y.shape[0]
    learning_rate = gamma / m
    for i in range(max_iters):
        score = np.dot(tx, initial_w)
        error = sigmoid_fn(score) - y  
        gradient = np.dot(tx.T, error)
        initial_w = initial_w - learning_rate * gradient

    cost = logistic_regression_cost(y, tx, initial_w) # imported from created_helpers

    return (initial_w, cost)
    

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    m = y.shape[0]
    learning_rate = gamma / m
    for i in range(max_iters):
        score = np.dot(tx, initial_w)
        error = sigmoid_fn(score) - y
        if (i == 0):
            gradient = np.dot(tx.T, error)
        else:
            gradient = np.dot(tx.T, error) + (lambda_ / m) * initial_w
        initial_w = initial_w - learning_rate * gradient

    cost = logistic_regression_cost(y, tx, initial_w) # imported from created_helpers

    return (initial_w, cost)
