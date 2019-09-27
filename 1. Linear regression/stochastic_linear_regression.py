import numpy
import math
import os
import math_helper

def compute(X, Y, w, params):
    learning_rate = params.learning_rate
    overfitting_penalty = params.overfitting_penalty
    max_gen = params.max_gen
    trace = params.trace
    output_file = params.output_file
    rows_per_gen = params.rows_per_gen
    
    if output_file is not None:
        #   Print initialization values
        output_file.write('Mode: stochastic\n')
        output_file.write('Learning rate: %f\n' % learning_rate)
        output_file.write('Number of rows per epoch: %f\n' % rows_per_gen)
        output_file.write('Initial w: %s\n' % w)
        output_file.write('Max generation: %d\n' % max_gen)
        output_file.write('\n')

        output_file.flush()
        os.fsync(output_file.fileno())    

    total = X.shape[0]
    gradient = [0] * X.shape[1]
    cur_gen = 0

    cur_index = 0
    while cur_gen < max_gen:
        mse = 0
        rmse = 0
        r_2 = 0
        
        #   Randomize if needed
        if cur_index + rows_per_gen >= total:
            full_frame = numpy.concatenate((X, Y.reshape(total,1)), axis=1)
            numpy.random.shuffle(full_frame)
            Y = full_frame[:, -1]
            X = full_frame[:, :-1]
            cur_index = 0
        gradient = [0] * X.shape[1]
        cur_gen += 1
        x = X[cur_index : cur_index + rows_per_gen]
        y = Y[cur_index : cur_index + rows_per_gen]
        gradient = math_helper.calculate_gradient(x, y, w, overfitting_penalty)

        if output_file is not None:
            (mse, rmse, r_2) = math_helper.calculate(X, Y, w)
            
            output_file.write('Generation: %d\n' % cur_gen)
            output_file.write('MSE: %f\n' % mse)
            output_file.write('RMSE: %f\n' % rmse)
            output_file.write('R2: %f\n' % r_2)
            output_file.write('Gradient: %s\n' % gradient)
            output_file.write('W: %s\n' % w)
            output_file.write('\n')

            #   Flush changes to file
            if cur_gen % 100 == 0:
                output_file.flush()
                os.fsync(output_file.fileno())

        if trace:
            if mse == 0 or rmse == 0 or r_2 == 0:
                (mse, rmse, r_2) = math_helper.calculate(X, Y, w)
            print("Gen: %5d | MSE: %5.5f | RMSE: %5.5f | R2: %5.5f" % (cur_gen, mse, rmse, r_2)) 

        gradient_dir = math_helper.get_direction(gradient)
        w = w - gradient_dir * learning_rate * (1 / math.log(cur_gen + 1))
        cur_index += rows_per_gen
    return w
