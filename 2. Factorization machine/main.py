import numpy
import random
import training_data_set_helper
from sklearn.model_selection import KFold

print('Restoring data for training..')

X = training_data_set_helper.restore_csr()
Y = training_data_set_helper.restore_df_target()

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
    E = numpy.ones((V.shape[1], 1))
    # Won't power up X because it consists only of ones and zeroes
    return b + X @ w + 0.5 * ((numpy.power(X @ V, 2) - X @ numpy.power(V, 2)) @ E)


def train(X, Y, step, batch_size, k, epochs_count):
    print('Starting training')
    print('Step: ', step)

    rows_count = X.shape[0]
    columns_count = X.shape[1]

    # Create V matrix
    V = numpy.random.normal(size=(columns_count, k))

    # Initialize weights
    b = 0
    w = numpy.random.normal(size=(columns_count, 1))

    batches_count = rows_count // batch_size + 1
    index = list(range(rows_count))
    rmse = 0

    for cur_epoch in range(epochs_count):
        if (cur_epoch != 0 and rmse < 1.02):
            break

        print('Current epoch: ' + str(cur_epoch))
        random.shuffle(index)

        for cur_batch in range(batches_count):
            has_enough_data_for_batch = (cur_batch + 1) * batch_size < X.shape[0]
            if (has_enough_data_for_batch):
                train_index = index[cur_batch * batch_size: (cur_batch + 1) * batch_size]
            else:
                train_index = index[cur_batch * batch_size:]

            X_batch = X[train_index]
            Y_batch = Y[train_index]

            new_Y_batch = compute_two_way_fm(X_batch, V, w, b)
            error = new_Y_batch - Y_batch
            error_x_2 = error * 2

            # Modify weights
            b -= numpy.mean(error_x_2) * step
            w -= numpy.mean(X_batch.multiply(error_x_2) * step, axis=0).reshape(-1, 1)

            # Modify V
            # Won't power up X because it consists only of ones and zeroes
            for f in range(k):
                arg_1 = X_batch.multiply(X_batch @ V[:, f].reshape(-1, 1))
                arg_2 = X_batch.multiply(V[:, f])
                arg_diff = (arg_1 - arg_2).multiply(error_x_2).mean(axis=0).T
                V[:, f] -= (numpy.asarray(arg_diff)).squeeze(axis=1)

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

    w, b, V = train(X_train, Y_train, step=1e-2, batch_size=4096, k=5, epochs_count=10)
    print('Training is completed')
    Y_trained = compute_two_way_fm(X_test, V, w, b)
    error = Y_trained - Y_test
    rmse = calculate_rmse(error)
    print('Final RMSE: ' + str(rmse))
    RMSE_test.append(rmse)