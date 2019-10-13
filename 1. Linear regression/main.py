import numpy
import pandas
import csv
from datetime import datetime

import math_helper
import features_handler
import simple_linear_regression
import stochastic_linear_regression
from math_helper import RegressionParameters

df = features_handler.export_training()
df_rand = features_handler.randomize(df)
df_features, df_target = features_handler.split_target(df)
features = df_features.values
target = df_target.values

mode = 'simple'
chunk_num = 5        
chunk_size = int(features.shape[0] / chunk_num)
params = RegressionParameters(
    learning_rate=0.24,
    regularization_param=0.005,
    min_grad_norm=0.1,
    max_gen=20000)

normalization_values = None
chosen_w_rmse = None
chosen_w = None

weights = [None] * chunk_num
r2_training = [0] * chunk_num
r2_validation = [0] * chunk_num
rmse_training = [0] * chunk_num
rmse_validation = [0] * chunk_num

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

    weights[validation_chunk] = w

    #   Calculate metrics for training set
    (mse, rmse, r2) = math_helper.calculate(X, Y, w)
    rmse_training[validation_chunk] = rmse
    r2_training[validation_chunk] = r2
    print("===\nTraining set metrics")
    print("MSE: %5.5f | RMSE: %5.5f | R2: %5.5f" % (mse, rmse, r2))

    (mse, rmse, r2) = math_helper.calculate(X_val, Y_val, w)
    rmse_validation[validation_chunk] = rmse
    r2_validation[validation_chunk] = r2
    print("Validation set metrics")
    print("MSE: %5.5f | RMSE: %5.5f | R2: %5.5f" % (mse, rmse, r2))
    print("W: %s" % w)

    if chosen_w_rmse is None or chosen_w_rmse > rmse:
        chosen_w_rmse = rmse
        chosen_w = w
        normalization_values = (x_min, x_range)

output_csv_name = 'output_' + datetime.now().strftime('%m-%d_%H-%M-%S') + '.csv'
with open(output_csv_name, 'wb+') as output_csv:
    csv_writer = csv.writer(output_csv)

    #   Print column names
    csv_writer.writerow(['', 'T1', 'T2', 'T3', 'T4', 'T5', 'E', 'STD'])

    csv_writer.writerow(numpy.concatenate((['R2 training'], r2_training, [numpy.mean(r2_training), numpy.std(r2_training)])))
    csv_writer.writerow(numpy.concatenate((['R2 validation'], r2_validation, [numpy.mean(r2_validation), numpy.std(r2_validation)])))
    csv_writer.writerow(numpy.concatenate((['RMSE training'], rmse_training, [numpy.mean(rmse_training), numpy.std(rmse_training)])))
    csv_writer.writerow(numpy.concatenate((['RMSE validation'], rmse_validation, [numpy.mean(rmse_validation), numpy.std(rmse_validation)])))
    weights_mean = numpy.mean(weights, axis=0)
    weights_std = numpy.std(weights, axis=0)

    for index in range(weights[0].shape[0]):
        csv_writer.writerow(['W'+str(index+1),
                             weights[0][index],
                             weights[1][index],
                             weights[2][index],
                             weights[3][index],
                             weights[4][index],
                             weights_mean[index],
                             weights_std[index]])
