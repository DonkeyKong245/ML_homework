import numpy
import math
import math_helper


def compute(X, Y, w, params):
    learning_rate = params.learning_rate
    regularization_param = params.regularization_param
    max_gen = params.max_gen
    rows_per_gen = params.rows_per_gen

    total = X.shape[0]
    cur_gen = 0

    cur_index = 0
    while cur_gen < max_gen:

        #   Randomize if needed
        if cur_index + rows_per_gen >= total:
            full_frame = numpy.concatenate((X, Y.reshape(total, 1)), axis=1)
            numpy.random.shuffle(full_frame)
            Y = full_frame[:, -1]
            X = full_frame[:, :-1]
            cur_index = 0

        cur_gen += 1
        x = X[cur_index: cur_index + rows_per_gen]
        y = Y[cur_index: cur_index + rows_per_gen]
        gradient = math_helper.calculate_gradient(x, y, w, regularization_param)
        gradient_dir = math_helper.get_direction(gradient)
        w = w - gradient_dir * learning_rate * (1 / math.log(cur_gen + 1))
        cur_index += rows_per_gen

    return w
