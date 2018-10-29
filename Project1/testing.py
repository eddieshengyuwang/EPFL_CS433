from implementation import *
from created_helpers import *
from proj1_helpers import *

y_train, x_train, ids_train = load_csv_data("train.csv")
y_test, x_test, ids_test = load_csv_data("test.csv")

def standardize(x):
    ''' fill your code in here...
    '''
    centered_data = x - np.mean(x, axis=0)
    std_data = centered_data / np.std(centered_data, axis=0)
    return std_data

x_train2 = standardize(x_train)
initial_w = np.zeros((x_train2.shape[1],1))
max_iters = 1000
gamma = 0.0000004

w_log, loss_log = logistic_regression(y_train, x_train2, initial_w, max_iters, gamma)
print(loss_log)