import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import helper
import collections
import os
import json

#Program to read the input gesture data and convert to univariate word vectors


class WordVectorIndex:
    def __init__(self, gesture_id, sensor_id, time):
        self.gesture_id = gesture_id
        self.sensor_id = sensor_id
        self.time = time

    def __hash__(self):
        return hash((self.gesture_id, self.sensor_id, self.time))

    def __equal__(self, other):
        return (self.gesture_id, self.sensor_id, self.time) == (other.gesture_id, other.sensor_id, other.time)

    def __not_equal__(self, other):
        return not(self == other)

    def __str__(self):
        return "{" + str(self.gesture_id) + "," + str(self.sensor_id) + "," + str(self.time) + "}"

# Loading parameters from the 'param.json' file
def load_params():
    with open("param.json") as config:
        data = json.load(config)
    return data

# Loading gestures from the given dataset
def load_data(dir):
    complete_df = pd.DataFrame()

    for filename in os.listdir(dir):
        if filename.endswith(".csv"): 
            df = pd.read_csv(dir + "/" + filename, header=None)

            sensor_id = list(range(1, len(df)+1))
            gesture_id = [filename[:-4]] * len(df)

            df['sensor_id'] = sensor_id
            df['gesture_id'] = gesture_id

            complete_df = pd.concat([complete_df, df])

    return complete_df

# Get the maximum and minimum values in the dataset for normalizing the values
def max_min_df(df):
    max_min = pd.DataFrame(columns = ['sensor_id', 'max', 'min'])
    sensor_ids = df.sensor_id.unique()

    for id in sensor_ids:
        sensor_df = df.loc[df['sensor_id'] == id]
        just_sensor_value_df = sensor_df.drop(['sensor_id', 'gesture_id'], axis=1)

        max_sensor_value = just_sensor_value_df.max(axis = 0).max()
        min_sensor_value = just_sensor_value_df.min(axis = 0).min()

        max_min = max_min.append({'sensor_id': str(id), 'max': max_sensor_value, 'min': min_sensor_value}, ignore_index=True)

    max_min = max_min.set_index(['sensor_id'])
    return max_min

# Normalize the values in the dataset
def normalize(row, max_min):
    sensor_id = row.sensor_id
    max_value = max_min.loc[[str(sensor_id)],['max']].values[0][0]
    min_value = max_min.loc[[str(sensor_id)],['min']].values[0][0]

    for i in row.index:
        if i == 'sensor_id' or i == 'gesture_id' or pd.isnull(row[i]):
            continue

        row[i] = (row[i] - min_value)/(max_value - min_value)
        row[i] = row[i] * 2 + (-1)

    return row

# Quantize the values in the dataset
def quantize(row, interval):
    sensor_id = row.sensor_id

    for i in row.index:
        if i == 'sensor_id' or i == 'gesture_id' or pd.isnull(row[i]):
            continue

        for key, value in interval.items():
            if row[i] >= value[0] and row[i] <= value[1]:
                row[i] = int(key)
                break

    return row

def generate_word_vectors(row, word_vectors, window_length, shift_length):
    sensor_id = row.sensor_id
    gesture_id = row.gesture_id

    row = row.drop(labels=['sensor_id', 'gesture_id'])

    i=0
    while i < (len(row.index)-window_length):
        if pd.isnull(row[i]):
            break

        temp_key = WordVectorIndex(gesture_id=gesture_id, sensor_id=sensor_id, time=i)
        # counter = collections.Counter()
        k = i
        temp_list = []

        while k < (i + window_length):
            if pd.isnull(row[k]):
                break

            temp_list.append(int(row[k]))
            k += 1

        if len(temp_list) < window_length:
            break

        # counter.update(temp_list)
        word_vectors[temp_key] = temp_list
        i += shift_length

    return row

if __name__ == "__main__":
    print("Starting Task 1:")

    print("Loading parameters..........")
    data = load_params()
    interval = helper.get_intervals(data['resolution'])

    print("Loading data from dataset........")
    df = load_data(data['directory'])
    max_min = max_min_df(df)

    print("Normalizing data.........")
    df = df.apply(lambda x: normalize(x, max_min), axis=1)
    df.to_csv('../Outputs/normalized.csv', index=False)

    print("Quantizing normalized data........")
    df = df.apply(lambda x: quantize(x, interval), axis=1)
    df.to_csv('../Outputs/quantized.csv', index=False)

    print("Generating words......")
    word_vectors = {}
    df.apply(lambda x: generate_word_vectors(x, word_vectors, data['window_length'], data['shift_length']), axis = 1)

    f = open("Extras/word_vector_dictionary.txt", "w")
    for k, v in word_vectors.items():
        f.write(str(k) + " " + str(v) + "\n")

    print("Creating word vector file for each gesture........")
    l = []
    for k, v in word_vectors.items():
        w = str(k).split(",")
        gesture_id = w[0][1:]
        if gesture_id not in l:
            l.append(gesture_id)
            f = open(data['directory'] + "/" + gesture_id + ".wrd", "w")
            f.write(str(k) + " " + str(v) + "\n")
        else:
            f = open(data['directory'] + "/" + gesture_id + ".wrd", "a")
            f.write(str(k) + " " + str(v) + "\n")

    f.close()
    print("Task 1 completed")
