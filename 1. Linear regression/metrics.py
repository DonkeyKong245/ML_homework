import numpy

def calculate(X, Y, w):
    total = Y.size
    y_var = Y.var()

    error = calculate_error(X, Y, w)
    loss = (error ** 2).sum(axis=0) / total
    r_2 = 1 - loss / y_var
    rmse = numpy.sqrt(loss)

    return (loss, r_2, rmse)


def calculate_error(X, Y, w):
    return Y - numpy.dot(X, w)


def calculate_gradient(X, Y, w):
    error = calculate_error(X, Y, w)
    return numpy.dot(numpy.transpose(X), error) * (-2 / X.shape[0])
