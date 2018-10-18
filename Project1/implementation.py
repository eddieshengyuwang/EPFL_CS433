'''
NOTE that in our implementation, we use m to represent # rows
and n to represent # columns
'''

import numpy as np
from created_helpers import batch_iter, least_squares_cost, sigmoid_fn, \
                            logistic_regression_cost, \
                            reg_logistic_regression_cost

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
    # equation is (X.T*X)^(-1) * (X.T*y)
    # question for TA's, should we use np.linalg.solve instead of invert?
    
    first_term = np.linalg.inv(np.dot(tx.T,tx)) # inv((n,m)x(m,n)) -> (n,n)
    second_term = np.dot(tx.T,y) # (n,m)x(m,1) -> (n,1)
    return np.dot(first_term, second_term) # (n,n)x(n,1) -> (n,1)

def ridge_regression(y, tx, lambda_):
    # equation is (X.T*X + (lambda')*I)^(-1) * (X.T*y)
    # where lambda' = 2*m*lambda_
    # question for TA's, should we use np.linalg.solve instead of invert?

    m = tx.shape[0]
    lambda_prime = 2*m*lambda_ # int

    xT_x = np.dot(tx.T, tx) # (n,m)x(m,n) -> (n,n)
    lambda_I = np.dot(lambda_prime, np.eye(n)) # int * (n,n) -> (n,n)
    xT_y = np.dot(tx.T, y) # (n,m)x(m,1) -> (n,1)

    return np.dot(np.linalg.inv(xT_x + lambda_I),xT_y) # (n,n)x(n,1) -> (n,1)


        
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    m = y.shape[0]
    learning_rate = gamma / m
    for i in max_iters:
        score = np.dot(tx, initial_w)
        error = sigmoid_fn(score) - y  
        gradient = np.dot(tx.T, error)
        initial_w = initial_w - learning_rate * gradient

    cost = logistic_regression_cost(y, tx, initial_w)

    return (initial_w, cost)
    

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    m = y.shape[0]
    learning_rate = gamma / m
    for i in max_iters:
        score = np.dot(tx, initial_w)
        error = sigmoid_fn(score) - y
        if (i == 0):
            gradient = np.dot(tx.T, error)
        else:
            gradient = np.dot(tx.T, error) + (lambda_ / m) * initial_w
        initial_w = initial_w - learning_rate * gradient

    cost = logistic_regression_cost(y, tx, initial_w)

    return (initial_w, cost)