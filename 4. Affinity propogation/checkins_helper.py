import numpy as np
import matplotlib.pyplot as plt

checkins_path = 'Dataset/total_checkins.txt'


def count_checkins():
    return 6442892
    
    counter = 0
    with open(checkins_path, 'r') as checkins_file:
        for line in checkins_file:
            counter += 1
            
    return counter


def get_checkins():
    line_index = 0    
    checkins_count = count_checkins()
    checkins = np.zeros((checkins_count, 2), dtype=np.int32)
    
    with open(checkins_path, 'r') as checkins_file:
        for line in checkins_file:
            line_split = line.split('\t')
            checkins[line_index, 0] = int(line_split[0])
            checkins[line_index, 1] = int(line_split[4])
            line_index += 1
            
    return checkins


def get_users(checkins):
    return np.unique(checkins[:, 0])
