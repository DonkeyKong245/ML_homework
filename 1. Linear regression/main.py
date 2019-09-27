import math
import time
import os
import numpy
import pandas
from datetime import datetime

import math_helper
import features_handler
import simple_linear_regression
import stochastic_linear_regression
from math_helper import regression_parameters

df = features_handler.export_training()
df_rand = features_handler.randomize(df)
df_features, df_target = features_handler.split_target(df)
features = df_features.values
target = df_target.values

mode = 'stochastic'
output_dir = 'Output'
output_file_name = '1_' + mode + '_'                            \
                   + datetime.now().strftime('%m-%d_%H-%M-%S')  \
                   + '.txt'
output_file_path = os.path.join(output_dir, output_file_name)
output_file = open(output_file_path, 'w+')
chunk_num = 5        
chunk_size = int(features.shape[0] / chunk_num)
params = regression_parameters(\
    learning_rate=0.6,\
    overfitting_penalty=None,\
    rows_per_gen=100,\
    max_gen=5000,\
    trace=False,\
    output_file=None)

normalization_values = None
chosen_w_rmse = None
chosen_w = None

#   5-cross validation
for validation_chunk in range(chunk_num):
    X = pandas.DataFrame(columns=df_features.columns)
    X_val = pandas.DataFrame(columns=df_features.columns)
    Y, Y_val = [], []
    
    for chunk in range(chunk_num):
            start_index = chunk * chunk_size
            end_index = (chunk + 1) * chunk_size
            new_chunk = df_features[start_index : end_index].copy()
            new_target_chunk = df_target[start_index : end_index].values.copy()
            if chunk == validation_chunk:
                X_val = X_val.append(new_chunk)
                Y_val = numpy.concatenate((Y_val, new_target_chunk), axis=0)
            else:
                X = X.append(new_chunk)
                Y = numpy.concatenate((Y, new_target_chunk), axis=0)            

    X = X.astype(numpy.float64).reset_index(drop=True).values
    X_val = X_val.astype(numpy.float64).reset_index(drop=True).values

    #   Normalize sets    
    (x_min, x_range) = features_handler.get_normalization_values(X)

    X = features_handler.normalize(X, x_min, x_range)
    b_column = numpy.transpose(numpy.array([[1] * X.shape[0]]))
    X = numpy.concatenate((X, b_column), axis=1)

    X_val = features_handler.normalize(X_val, x_min, x_range)
    b_column = numpy.transpose(numpy.array([[1] * X_val.shape[0]]))
    X_val = numpy.concatenate((X_val, b_column), axis=1)
    
    w = numpy.random.rand(X.shape[1])
    
    if mode == 'stochastic':
        w = stochastic_linear_regression.compute(X, Y, w, params)
    else:
        if mode == 'simple':
            w = simple_linear_regression.compute(X, Y, w, params)
        else:
            raise Exception('Invalid mode!')

    #   Calculate metrics for training set
    (mse, rmse, r2) = math_helper.calculate(X, Y, w)
    print("===\nTraining set metrics")
    print("MSE: %5.5f | RMSE: %5.5f | R2: %5.5f" % (mse, rmse, r2))
    output_file.write('Training set metrics\n')
    output_file.write('MSE: %f\n' % mse)
    output_file.write('RMSE: %f\n' % rmse)
    output_file.write('R2: %f\n' % r2)
    output_file.flush()
    os.fsync(output_file.fileno())   

    (mse, rmse, r2) = math_helper.calculate(X_val, Y_val, w)
    print("Validation set metrics")
    print("MSE: %5.5f | RMSE: %5.5f | R2: %5.5f" % (mse, rmse, r2))
    print("W: %s" % w)
    output_file.write('Validation set metrics\n')
    output_file.write('MSE: %f\n' % mse)
    output_file.write('RMSE: %f\n' % rmse)
    output_file.write('R2: %f\n' % r2)
    output_file.flush()
    os.fsync(output_file.fileno())

    if chosen_w_rmse is None or chosen_w_rmse > rmse:
        chosen_w_rmse = rmse
        chosen_w = w
        normalization_values = (x_min, x_range)
        

df_test = features_handler.export_testing()
X, Y = features_handler.split_target(df_test)
X = X.astype(numpy.float64).reset_index(drop=True).values
Y = Y.astype(numpy.float64).reset_index(drop=True).values
X = features_handler.normalize(X, normalization_values[0], normalization_values[1])
b_column = numpy.transpose(numpy.array([[1] * X.shape[0]]))
X = numpy.concatenate((X, b_column), axis=1)

(mse, rmse, r2) = math_helper.calculate(X, Y, chosen_w)
print("===\nTest set metrics")
print("MSE: %5.5f | RMSE: %5.5f | R2: %5.5f" % (mse, rmse, r2))
print("W: %s" % chosen_w)
output_file.write('Test set metrics\n')
output_file.write('MSE: %f\n' % mse)
output_file.write('RMSE: %f\n' % rmse)
output_file.write('R2: %f\n' % r2)
output_file.flush()
os.fsync(output_file.fileno())

