import csv
import sys
import os.path
import numpy
import re
import pandas
from io import StringIO
from scipy import sparse
from sklearn.externals import joblib
from sklearn.model_selection import KFold
from datetime import datetime

data_set_folder = "Dataset"
training_data_name_format = "combined_data_"
training_combined_data_name = "combined_data.txt"
df_prefix = "df_"
target_keyword = "target.bin"
csr_file_name = "csr.dump"
stats_name_format = "stats.csv"

first_date_str = "1999-11-11"
first_date = datetime.strptime(first_date_str, "%Y-%m-%d")
total_rows_in_df = 100480507
max_user_id = 2649429
max_film_id = 17770
month_to_days_passed = {1: 0,
                        2: 31,
                        3: 31 + 28,
                        4: 31 + 28 + 31,
                        5: 31 + 28 + 31 + 30,
                        6: 31 + 28 + 31 + 30 + 31,
                        7: 31 + 28 + 31 + 30 + 31 + 30,
                        8: 31 + 28 + 31 + 30 + 31 + 30 + 31,
                        9: 31 + 28 + 31 + 30 + 31 + 30 + 31 + 31,
                        10: 31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30,
                        11: 31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31,
                        12: 31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31 + 30,
                        13: 31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31 + 30 + 31}


def get_days_diff(date_to, z_year=1999, z_month=11, z_day=11):
    year, month, day = [int(val) for val in date_to.strip().split('-')]

    if z_year != year:
        #   month_to_days_passed[13] - (month_to_days_passed[z_month] + (z_day - 1))
        days_from_z_date_to_2000_year = 51
        days_passed_since_year_started = month_to_days_passed[month] + (day - 1)
        years_diff = year - z_year

        return days_from_z_date_to_2000_year + \
               (years_diff - 1) * month_to_days_passed[13] + \
               days_passed_since_year_started
    else:
        return month_to_days_passed[month] - \
               month_to_days_passed[z_month] - \
               z_day + 1 + \
               (day - 1)


def list_all_training_file_names():
    return [os.path.join(data_set_folder, f) for f in os.listdir(data_set_folder)
            if os.path.isfile(os.path.join(data_set_folder, f))
            and training_data_name_format in f]


def combine_data():
    with open(training_combined_data_name, 'w') as combined_data_file:
        file_names = list_all_training_file_names()

        for file_name in file_names:
            with open(file_name) as file:
                data = file.read()
                combined_data_file.write(data)


def get_film_ids():
    film_ids = []
    file_names = list_all_training_file_names()

    for file_name in file_names:
        with open(file_name) as file:
            text = file.read()
            film_ids += list(map(lambda x: int(x[:-1]), re.findall(r'\d+:', text)))

    return film_ids


def get_user_ids():
    user_ids = []
    file_names = list_all_training_file_names()

    for file_name in file_names:
        with open(file_name) as file:
            text = file.read()
            user_ids += list(map(lambda x: int(x.split(',')[0]), re.findall(r'\d+,\d+,\d+', text)))

    #   Make unique list
    user_ids = list(set(user_ids))

    #   Sort the list
    user_ids.sort()
    return user_ids


#   Max user id is 2649429
def get_max_user_id():
    return max_user_id

    user_ids = get_user_ids()
    new_max_user_id = max(user_ids)
    return new_max_user_id


#   Max film id is 17770
def get_max_film_id():
    return max_film_id

    film_ids = get_film_ids()
    new_max_film_id = max(film_ids)
    return new_max_film_id


#   First date is 1999-11-11
def get_first_date():
    return datetime.strptime(first_date_str, "%Y-%m-%d")

    first_date = None
    file_names = list_all_training_file_names()

    for file_name in file_names:
        with open(file_name) as file:
            for line in file:
                items = line.split(',')

                if len(items) == 3:
                    date_time_str = items[2]
                    date_time = datetime.strptime(date_time_str, "%Y-%m-%d\n")
                    if first_date is None or first_date > date_time:
                        first_date = date_time
    return first_date


def count_number_of_data_rows(file_name):
    count = 0
    with open(file_name) as file:
        for line in file:
            if not line.__contains__(":"):
                count += 1

    return count


def get_user_to_film_ratings():
    unsorted_user_to_rated_films = dict()
    file_names = list_all_training_file_names()

    for file_name in file_names:
        print ("Processing %s file" % file_name)

        film_id = -1
        with open(file_name) as file:
            for line in file:
                items = line.split(',')
                if len(items) == 3 and film_id != -1:
                    user_id = int(items[0])
                    days_diff = get_days_diff(items[2])

                    if not unsorted_user_to_rated_films.has_key(user_id):
                        unsorted_user_to_rated_films[user_id] = []

                    if film_id not in [ex_film_rating[0] for ex_film_rating in unsorted_user_to_rated_films[user_id]]:
                        unsorted_user_to_rated_films[user_id].append([film_id, days_diff])
                else:
                    items = line.split(':')
                    if len(items) == 2:
                        film_id = int(items[0])

    user_to_ordered_rated_films = dict()
    for user_id in unsorted_user_to_rated_films:
        user_to_ordered_rated_films[user_id] = [rated_film[0]
                                                for rated_film
                                                in sorted(unsorted_user_to_rated_films[user_id],
                                                          key=lambda rated_film: rated_film[1])]

    return user_to_ordered_rated_films


def stats_to_csv():
    output_csv_name = stats_name_format
    with open('unsorted' + output_csv_name, 'wb+') as output_csv:
        csv_writer = csv.writer(output_csv)
        user_to_ordered_rated_films_collection = get_user_to_film_ratings()

        for user_id in user_to_ordered_rated_films_collection:
            user_to_ordered_rated_films = user_to_ordered_rated_films_collection[user_id]
            csv_writer.writerow(numpy.concatenate(([user_id],
                                                  user_to_ordered_rated_films)))


def sort_stats_csv():
    with open('unsorted' + stats_name_format, 'r') as unsorted_stats_csv:
        csv_reader = csv.reader(unsorted_stats_csv)
        unsorted_data = list(csv_reader)
        sorted_data = sorted(unsorted_data, key=lambda row: int(row[0]))

        with open(stats_name_format, 'wb+') as output_csv:
            csv_writer = csv.writer(output_csv)

            for row in sorted_data:
                csv_writer.writerow(row)


def get_user_to_rated_films():
    with open(stats_name_format, 'r') as output_csv:
        csv_reader = csv.reader(output_csv)

        user_to_rated_films = dict()
        for row in csv_reader:
            key = int(row[0])
            rated_films = [int(film_id) for film_id in row[1:]]
            user_to_rated_films[key] = rated_films

        return user_to_rated_films


def read_df():
    df = []
    with open(os.path.join(data_set_folder, training_combined_data_name)) as data_file:
        text = data_file.read()

        film_ids = list(map(lambda x: int(x[:-1]), re.findall(r'\d+:', text)))
        df_raw = re.split(r'\d+:', text)
        df_raw = df_raw[1:]

        for df_raw_row, film_id in zip(df_raw, film_ids):
            str = StringIO(df_raw_row)
            sub_df = pandas.read_csv(str, names=['UserID', 'Rate', 'Date'])
            sub_df['FilmID'] = film_id
            df.append(sub_df)

        df = pandas.concat(df)
    return df


def get_df_target(df):
    return df.Rate.values.T


def dump_df():
    df = read_df()
    total_rows = len(df.index)
    max_rows_per_dump = 25000000

    rows_done = 0
    cur_file_index = 0
    while rows_done < total_rows:
        file_name = (data_set_folder + "\\" + 
                    df_prefix + str(cur_file_index) + ".bin")

        rows_to_dump = max_rows_per_dump
        if total_rows - rows_done < max_rows_per_dump:
            rows_to_dump = total_rows - rows_done

        joblib.dump(df[cur_file_index : cur_file_index + rows_to_dump], file_name)
        rows_done += rows_to_dump
        cur_file_index += 1


def restore_df():
    df_file_names = [os.path.join(data_set_folder, f) for f in os.listdir(data_set_folder)
                    if os.path.isfile(os.path.join(data_set_folder, f))
                    and df_prefix in f]

    df = []
    df_file_names.sort()

    for df_file_name in df_file_names:
        sub_df = joblib.load(df_file_name)
        df.append(sub_df)

    df = pandas.concat(df)
    df = df.reset_index(drop=True)
    return df


def restore_csr():
    csr_file_path = os.path.join(data_set_folder, csr_file_name)
    csr = joblib.load(csr_file_path)
    return csr.astype(numpy.float32)


def restore_csr_target():
    df_target = joblib.load(os.path.join(data_set_folder, df_prefix + target_keyword))
    return sparse.csr_matrix(df_target, dtype=numpy.float32)


def df_to_csr(df):
    rows_count = len(df.index)
    columns_count = (max_user_id + 1
                     + max_film_id + 1)

    cur_row = 0
    lil = sparse.lil_matrix((rows_count, columns_count))

    for i in range(rows_count):
        lil[cur_row, df.UserID[i]] = 1
        lil[cur_row, (max_user_id + 1) + df.FilmID[i]] = 1
        cur_row += 1

    csr = sparse.csr_matrix(lil)
    csr = csr.astype('B', copy=True)
    return csr