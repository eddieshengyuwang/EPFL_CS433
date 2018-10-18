import numpy as np

def least_squares_cost(y, tx, w):
    '''
    :param y: np.array (m,1)
    :param tx: np.array (m,n)
    :param w: np.array (n,1)
    :return: float
    '''
    cost = (1/2) * np.sum(np.square(y-np.dot(tx, w)))
    return cost
 
def sigmoid_fn(z):
    h = 1 / (1 + np.exp(-z))
    return h
    
def logistic_regression_cost(y, tx, w):
    z = np.dot(tx, w)
    h = sigmoid_fn(z)
    cost  = (-y) * np.log(h) - (1 - y) * np.log(1 - h)
    return cost
   
    
def reg_logistic_regression_cost(y, tx, w, lambda_):
    cost_1 = logistic_regression_cost(y, tx, w)
    cost_2 = (lambda_ / 2 * len(tx)) * \
              np.sum(np.power(w[:,1:w.shape[1]], 2))
    cost = cost_1 + cost_2
    return cost
    
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True): # taken from ex02 helper.py
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
