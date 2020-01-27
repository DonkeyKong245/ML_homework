import numpy
import random
import pandas
import scipy.sparse as scipy
import training_data_set_helper
from sklearn.model_selection import KFold

print('Restoring data for training..')

X = training_data_set_helper.restore_csr()
Y = training_data_set_helper.restore_csr_target()

print('Data for training successfully restored')


# @jit(nopython=True)
def calculate_error(X, Y, w, b):
    return Y - (numpy.dot(X, w) + b)


# @jit(nopython=True)
def calculate_rmse(error):
    mse = (numpy.power(error, 2)).sum(axis=0) / error.size
    return numpy.sqrt(mse)


# @jit(nopython=True)
def compute_two_way_fm(X, V, w, b):
    return b + X * w + 0.5 * ((X * V).power(2) - X.power(2) * V.power(2)).sum(axis=1)


def train(X, Y, step, batch_size, k, epochs_count):
    print('Starting training')
    print('Step: ', step)

    rows_count = X.shape[0]
    columns_count = X.shape[1]

    # Create V matrix
    V = scipy.csr_matrix(numpy.random.normal(size=(columns_count, k)) * 1e-15)

    # Initialize weights
    b = 0
    w = numpy.random.normal(size=(columns_count, 1)) * 1e-15

    batches_count = rows_count // batch_size + 1
    rmse = 0

    for cur_epoch in range(epochs_count):
        if (cur_epoch != 0 and rmse < 1.02):
            break

        print('Current epoch: ' + str(cur_epoch))
        print('Shuffling X..')

        random_index = numpy.arange(rows_count)
        numpy.random.shuffle(random_index)

        X = X[random_index, :]
        Y = Y[random_index]

        print('Shuffled')

        for cur_batch in range(batches_count):
            has_enough_data_for_batch = (cur_batch + 1) * batch_size < X.shape[0]
            if (has_enough_data_for_batch):
                X_batch = X[cur_batch * batch_size : (cur_batch + 1) * batch_size]
                Y_batch = Y[cur_batch * batch_size : (cur_batch + 1) * batch_size]
            else:
                X_batch = X[cur_batch * batch_size :]
                Y_batch = Y[cur_batch * batch_size :]

            new_Y_batch = compute_two_way_fm(X_batch, V, w, b)
            error = Y_batch - new_Y_batch
            Y_batch_diff = -2 * error / batch_size

            # Modify b
            b -= Y_batch_diff.sum() * step

            # Modify w
            X_transposed = X_batch.transpose()
            w -= X_batch.transpose().dot(Y_batch_diff) * step

            # Modify V
            X_x_V = X_batch.dot(V)
            arg1 = X_x_V.multiply(Y_batch_diff.reshape(-1, 1))
            arg1 = X_transposed.dot(arg1)

            X_transposed_squared = X_transposed.power(2)
            subarg_csr = scipy.csr_matrix(X_transposed_squared.dot(Y_batch_diff))
            arg2 = V.multiply(subarg_csr)

            V -= (arg1 - arg2) * step

            if (cur_batch % 200 == 0):
                rmse = calculate_rmse(error)
                print('Batch number: ' + str(cur_batch) + '\t| RMSE: ' + str(rmse))

        next_epoch = cur_epoch + 1
        step *= numpy.exp(-next_epoch * 0.01)
    return w, b, V


RMSE_train, RMSE_test = [], []
kfold = KFold(n_splits=5)

for n_train, n_test in kfold.split(X):
    print('===')
    print('New split')
    X_train = X[n_train]
    X_test = X[n_test]
    Y_train = Y[n_train]
    Y_test = Y[n_test]

    w, b, V = train(X_train, Y_train, step=1e-1, batch_size=4096, k=3, epochs_count=10)

    print('Training is completed')
    Y_trained = compute_two_way_fm(X_test, V, w, b)
    error = Y_trained - Y_test
    rmse = calculate_rmse(error)

    print('Final RMSE: ' + str(rmse))
    RMSE_test.append(rmse)

results = pandas.DataFrame(numpy.vstack([RMSE_test]), index=['Test'])
results = pandas.concat([results, results.mean(axis=1).rename('Mean'), results.std(axis=1).rename('Std')], axis=1)