import os
import numpy
import pandas

dataset_folder_name = 'Dataset'
training_folder_name = 'Training'
testing_folder_name = 'Testing'
dataset_file_format = 'Features_Variant_'
csv_ext = '.csv'

dataset_file = os.path.join(dataset_folder_name,        \
                            training_folder_name,       \
                            dataset_file_format + '1' + csv_ext)

likes = 1
#   Page Popularity/likes
#   Decimal Encoding
#   Page feature
#   Defines the popularity or support for the source of the document.

checkins = 2
#   Page Checkins
#   Decimal Encoding
#   Page feature
#   Describes how many individuals so far visited this place. This feature is only associated with the places eg:some institution, place, theater etc.

interest = 3
#   Page talking about
#   Decimal Encoding
#   Page feature
#   Defines the daily interest of individuals towards source of the document/ Post. The people who actually come back to the page, after liking the page. This include activities such as comments, likes to a post, shares, etc by visitors to the page.

category = 4
#   Page Category
#   Value Encoding
#   Page feature
#   Defines the category of the source of the document eg: place, institution, brand etc.

derived = range(5, 30)
#   Derived
#   Decimal Encoding
#   Derived feature
#   These features are aggregated by page, by calculating min, max, average, median and standard deviation of essential features.

cc1 = 30
#   CC1
#   Decimal Encoding
#   Essential feature
#   The total number of comments before selected base date/time.

cc2 = 31
#   CC2
#   Decimal Encoding
#   Essential feature
#   The number of comments in last 24 hours, relative to base date/time.

cc3 = 32
#   CC3
#   Decimal Encoding
#   Essential feature
#   The number of comments in last 48 to last 24 hours relative to base date/time.

cc4 = 33
#   CC4
#   Decimal Encoding
#   Essential feature
#   The number of comments in the first 24 hours after the publication of post but before base date/time.

cc5 = 34
#   CC5
#   Decimal Encoding
#   Essential feature
#   The difference between CC2 and CC3.

base_time = 35
#   Base time
#   Decimal(0-71) Encoding
#   Other feature
#   Selected time in order to simulate the scenario.

p_length = 36
#   Post length
#   Decimal Encoding
#   Other feature
#   Character count in the post.

ps_count = 37
#   Post Share Count
#   Decimal Encoding
#   Other feature
#   This features counts the no of shares of the post, that how many peoples had shared this post on to their timeline.

pp_status = 38
#   Post Promotion Status
#   Binary Encoding
#   Other feature
#   To reach more people with posts in News Feed, individual promote their post and this features tells that whether the post is promoted(1) or not(0).

hours = 39
#   H Local
#   Decimal(0-23) Encoding
#   Other feature
#   This describes the H hrs, for which we have the target variable/ comments received.

pp_weakday = range(40,47)
#   Post published weekday
#   Binary Encoding
#   Weekdays feature
#   This represents the day(Sunday...Saturday) on which the post was published.

weakdays = range(47,53)
#   Base DateTime weekday
#   Binary Encoding
#   Weekdays feature
#   This represents the day(Sunday...Saturday) on selected base Date/Time.

target = 54
target_str = 'Target'
#   Target Variable
#   Decimal
#   Target
#   The no of comments in next H hrs(H is given in Feature no 39).

#   Import training data frame
def export():
    dataset_path = os.path.abspath(dataset_file)
    return pandas.read_csv(dataset_path,    \
                           header=None,     \
                           names=[str(i) for i in range(0, target-1)] + [target_str])

#   Randomize data frame
def randomize(df):
    #   frac=1 ~ sample must contain 100% of original data frame
    #   drop=True ~ delete old index column
    return df.sample(frac=1).reset_index(drop=True)

#   Exclude target from data frame
def split_target(df):
    df_target = df[target_str]
    df_features = df.drop(target_str, axis=1)
    return df_features, df_target

#   Normalize data frame
def normalize(df):
    df_mean = df.mean()
    df_var = df.var()
    df = (df - df_mean) / df_var
    return df.fillna(0)

#   Add additional column for b-parameter
def append_b(df):
    #   data=1 ~ use '1' as initial value
    #   index=.. ~ use custom index
    collection_b = pandas.Series(data=1, index=df.index)
    return df.assign(B=collection_b.values)
