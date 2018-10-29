from proj1_helpers import load_csv_data, predict_labels, create_csv_submission
import numpy as np
from datetime import datetime
from created_helpers import *

print("loading data")
y_train, x_train, ids_train = load_csv_data("train.csv")
y_test, x_test, ids_test = load_csv_data("test.csv")

# same ridge_regression as in implementations.py but just
# returning loss
def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    return np.linalg.solve(a, b)

# cross_validation code taken from lab
def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""
    # get k'th subgroup in test, others in train
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    y_te = y[te_indice]
    y_tr = y[tr_indice]
    x_te = x[te_indice]
    x_tr = x[tr_indice]
    # form data with polynomial degree
    tx_tr = build_poly(x_tr, degree)
    tx_te = build_poly(x_te, degree)
    # ridge regression
    w = ridge_regression(y_tr, tx_tr, lambda_)
    # calculate the loss for train and test data
    loss_tr = np.sqrt(2 * compute_mse(y_tr, tx_tr, w))
    loss_te = np.sqrt(2 * compute_mse(y_te, tx_te, w))
    return loss_tr, loss_te,w

degree = 6

# cross_validation_demo taken from lab
def cross_validation_demo():
    seed = 12
    k_fold = 5
    lambdas = np.logspace(-4, 0, 5)
    # split data in k fold
    k_indices = build_k_indices(y_train, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    # cross validation
    dict_lambda_weight = {}
    for ind, lambda_ in enumerate(lambdas):
        rmse_tr_tmp = []
        rmse_te_tmp = []
        for k in range(k_fold):
            loss_tr, loss_te,weight = cross_validation(y_train, x_train, k_indices, k, lambda_, degree)
            rmse_tr_tmp.append(loss_tr)
            rmse_te_tmp.append(loss_te)
            if lambda_ in dict_lambda_weight:
                if dict_lambda_weight[lambda_][0] > loss_te:
                    dict_lambda_weight[lambda_][1] = weight
            else:
                dict_lambda_weight[lambda_] = [loss_te, weight]
            rmse_tr_tmp.append(loss_tr)
        rmse_tr.append(np.mean(rmse_tr_tmp))
        rmse_te.append(np.mean(rmse_te_tmp))
        print("lambda={l:.3f}, Training RMSE={tr:.3f}, Testing RMSE={te:.3f}".format(
               l=lambda_, tr=rmse_tr[ind], te=rmse_te[ind]))
    ind_lambda_opt = np.argmin(rmse_te)
    best_lambda = lambdas[ind_lambda_opt]
    best_rmse = rmse_te[ind_lambda_opt]
    best_weight = dict_lambda_weight[best_lambda][1]
    return best_weight, best_rmse, best_lambda

print("training")
optimal_weight, best_rmse, best_lambda = cross_validation_demo()

x_train2 = build_poly(x_train, degree)
y_pred = predict_labels(optimal_weight, x_train2)

output = accuracy(y_pred, y_train)

print("done, training accuracy:")
print(output)

x_test2 = build_poly(x_test, degree)
y_pred2 = predict_labels(optimal_weight, x_test2)

print("creating submission")
create_csv_submission(ids_test, y_pred2,'ridge_regression_final.csv')
