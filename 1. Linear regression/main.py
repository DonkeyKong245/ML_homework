import math
import time
import os
import numpy
import pandas
from datetime import datetime

import metrics
import features_handler
import simple_linear_regression
import stochastic_linear_regression

full_output = False
mode = 'simple'
#mode = 'stochastic'
chunk_num = 5        

df = features_handler.export()
df_rand = features_handler.randomize(df)
df_features, df_target = features_handler.split_target(df_rand)
df_features = features_handler.normalize(df_features)
df_features = features_handler.append_b(df_features)

features = df_features.values
target = df_target.values

output_dir = 'Output'
output_file_name = '1_' + mode + '_'                            \
                   + datetime.now().strftime('%m-%d_%H-%M-%S')  \
                   + '.txt'
output_file_path = os.path.join(output_dir, output_file_name)
output_file = open(output_file_path, 'w+')

chunk_size = int(features.shape[0] / chunk_num)
for test_chunk in range(chunk_num):
    
    training_set = numpy.resize([[]],(0, 54))
    training_target_set = []
    for training_chunk in range(chunk_num):
        if training_chunk == test_chunk:
            continue
        
        start_index = training_chunk * chunk_size
        end_index = (training_chunk + 1) * chunk_size
        new_chunk = features[start_index : end_index].copy()
        new_target_chunk = target[start_index : end_index].copy()
        
        training_set = numpy.concatenate((training_set, new_chunk), axis=0)        
        training_target_set = numpy.concatenate((training_target_set, new_target_chunk), axis=0)
    
    learning_rate = 1.5
    eps = 0.001
    w = [1] * features.shape[1]

    if mode == 'stochastic':
        w = stochastic_linear_regression.compute(training_set, training_target_set, eps, learning_rate, w, output_file, 100, full_output)
    else:
        w = simple_linear_regression.compute(training_set, training_target_set, eps, learning_rate, w, output_file, full_output)               

    #   Calculate metrics for training set
    X = training_set
    Y = training_target_set

    (loss, r_2, rmse) = metrics.calculate(X, Y, w)

    print("===\nTraining set metrics")
    print("Loss: %5.2f | R2: %5.3f | RMSE: %.3f\n" % (loss, r_2, rmse))

    output_file.write('Training set metrics\n')
    output_file.write('Loss: %f\n' % loss)
    output_file.write('R2: %f\n' % r_2)
    output_file.write('RMSE: %f\n' % rmse)
    output_file.flush()
    os.fsync(output_file.fileno())   

    #   Calculate metrics for test set
    start_index = test_chunk * chunk_size
    end_index = (test_chunk + 1) * chunk_size
    X = features[start_index : end_index].copy()
    Y = target[start_index : end_index].copy()

    (loss, r_2, rmse) = metrics.calculate(X, Y, w)

    print("Test set metrics")
    print("Loss: %5.2f | R2: %5.3f | RMSE: %.3f\n" % (loss, r_2, rmse))
    output_file.write('Test set metrics\n')
    output_file.write('Loss: %f\n' % loss)
    output_file.write('R2: %f\n' % r_2)
    output_file.write('RMSE: %f\n' % rmse)
    output_file.flush()
    os.fsync(output_file.fileno())
