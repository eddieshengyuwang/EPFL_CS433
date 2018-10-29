'''
NOTE that in our implementation, we use m to represent # rows
and n to represent # columns
'''

import numpy as np
from created_helpers import *

def least_squares(y, tx):
    """Linear regresison using normal equations"""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    
    optimal_weight = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, optimal_weight) # imported from created_helpers
    
    return (optimal_weight, loss)

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    
    """Linear regression using gradient descent"""
    y = np.expand_dims(y, axis=1)
    w = initial_w  
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient(y, tx, w)
        loss = calculate_mse(err) 
        # gradient w by descent update
        w = w - gamma * grad
        
    return (w, loss)

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """ Linear regression using stochastic gradient descent""" 
    y = np.expand_dims(y, axis=1)
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            # compute a stochastic gradient and loss
            grad, _ = compute_stoch_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            loss = compute_loss(y, tx, w) 
            
    return (w, loss)

    
def ridge_regression(y, tx, lambda_):
    """implement ridge regression using normal equations."""
    
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    
    optimal_weight = np.linalg.solve(a, b)
    loss = ridge_regression_cost(y, tx, optimal_weight, lambda_)
    return (optimal_weight, loss)

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """ Logistic regression using gradient descent"""
    y = np.expand_dims(y, axis=1)  # assume that y is unchanged when loaded from data
    w = initial_w
    loss_prev = 0
    for n_iter in range(max_iters):
        loss = calculate_loss_logistic(y, tx, w)

        # convergence criteria
        if abs(loss_prev - loss) < 0.00001:
            break
        if np.isnan(loss):
            break
        print(loss, " ", loss_prev, " ", n_iter)
        gradient = calculate_gradient_logistic(y, tx, w)
        w = w - gamma * gradient
        loss_prev = loss
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent"""
    y = np.expand_dims(y, axis=1)
    w = initial_w
    loss_prev = 0
    for n_iter in range(max_iters):
        loss = calculate_loss_logistic(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))

        # convergence criteria
        if abs(loss_prev - loss) < 0.00001:
            break
        if np.isnan(loss):
            break
        print(loss, " ", loss_prev, " ", n_iter)
        gradient = calculate_gradient_logistic(y, tx, w) + 2 * lambda_ * w
        w -= gamma * gradient
        loss_prev = loss
    return w, loss