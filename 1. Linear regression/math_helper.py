import numpy


class RegressionParameters:
    def __init__(self,
                 learning_rate,
                 regularization_param=None,
                 rows_per_gen=1,
                 max_gen=5000,
                 min_grad_norm=0.1,
                 trace=False,
                 output_file=None):
        self.learning_rate = learning_rate
        self.regularization_param = regularization_param
        self.rows_per_gen = rows_per_gen
        self.max_gen = max_gen
        self.trace = trace
        self.output_file = output_file
        self.min_grad_norm = min_grad_norm


def calculate(X, Y, w):
    error = calculate_error(X, Y, w)
    (mse, rmse) = calculate_mse_and_rmse(error)
    r_2 = calculate_r2(error, Y)
    return (mse, rmse, r_2)


def calculate_mse_and_rmse(error):
    mse = (error ** 2).sum(axis=0) / error.size
    return (mse, numpy.sqrt(mse))


def calculate_r2(error, Y):
    y_mean = Y.mean()
    return 1 - ((error ** 2).sum(axis=0) / ((Y - y_mean) ** 2).sum(axis=0))


def calculate_error(X, Y, w):
    return Y - numpy.dot(X, w)


def calculate_gradient(X, Y, w, regularization_param=None):
    error = calculate_error(X, Y, w)
    gradient = -2 * numpy.dot(numpy.transpose(X), error) / X.shape[0]
    if regularization_param is not None:
        regularization_value = 2 * regularization_param * w
        gradient += regularization_value
    return gradient


def get_direction(v):
    norm = numpy.linalg.norm(v)
    return numpy.asarray([x / norm for x in v])
