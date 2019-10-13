import math
import math_helper
import numpy

def compute(X, Y, w, params):
    learning_rate = params.learning_rate
    regularization_param = params.regularization_param
    max_gen = params.max_gen
    min_grad_norm = params.min_grad_norm

    gradient = [1] * X.shape[1]
    cur_gen = 0
    while cur_gen < max_gen and numpy.linalg.norm(gradient) > min_grad_norm:
        cur_gen += 1
        gradient = math_helper.calculate_gradient(X, Y, w, regularization_param)
        gradient_dir = math_helper.get_direction(gradient)
        w = w - gradient_dir * learning_rate * (1 / math.log(cur_gen + 1))
    return w
