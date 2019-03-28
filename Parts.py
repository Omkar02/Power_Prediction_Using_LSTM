#----------------------------------------loading file
from get_val_from_ubidots import val_sender_ubi,get_val_ubi
import numpy as np
import csv

def load_data(dataset_path, sequence_length=30, prediction_steps=5, ratio_of_data=1.0):
    # max_values = ratio_of_data * 2075259  # 2075259 is the total number of measurements from Dec 2006 to Nov 2010

    # Load data from file
    with open(dataset_path) as file:
        data_file = csv.reader(file, delimiter=";")
        power_consumption = []
        number_of_values = 0
        for line in data_file:
            # print(line)
            try:
                power_consumption.append(float(line[2]))
                number_of_values += 1
            except ValueError:
                pass
            # if number_of_values >= max_values:  # limit data to be considered by model according to max_values
            #     break

    print('Loaded data from csv.')
    windowed_data = []
    # Format data into rolling window sequences
    for index in range(len(power_consumption) - sequence_length):  # for e.g: index=0 => 123, index=1 => 234 etc.
        windowed_data.append(power_consumption[index: index + sequence_length])
    windowed_data = np.array(windowed_data)  # shape (number of samples, sequence length)

    # Center data
    data_mean = windowed_data.mean()
    windowed_data -= data_mean
    print('Center data so mean is zero (subtract each data point by mean of value: ', data_mean, ')')
    print('Data  : ', windowed_data.shape)

    train_set_ratio = 0.9
    row = int(round(train_set_ratio * windowed_data.shape[0]))
    # print(windowed_data[0])
    train = windowed_data[:row, :]
    x_test = windowed_data[row:, :-prediction_steps]
    y_test = windowed_data[row:, -prediction_steps:]
    data_read = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

    return [data_read, data_mean]
# print(load_data('data/household_power_consumption.txt'))

#--------------------------------------------------------LSTM
import time
from keras.layers.recurrent import LSTM
from keras.layers.core import Activation, Dense, Dropout
from keras.models import Sequential, load_model
from numpy.random import seed
seed(1234)  # seed random numbers for Keras
from tensorflow import set_random_seed
set_random_seed(2)  # seed random numbers for Tensorflow backend

def build_model(prediction_steps):
    model = Sequential()
    layers = [1, 75, 100, prediction_steps]
    model.add(LSTM(layers[1], input_shape=(None, layers[0]), return_sequences=True))  # add first layer
    model.add(Dropout(0.2))  # add dropout for first layer
    model.add(LSTM(layers[2], return_sequences=False))  # add second layer
    model.add(Dropout(0.2))  # add dropout for second layer
    model.add(Dense(layers[3]))  # add output layer
    model.add(Activation('linear'))  # output layer with linear activation
    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print('Compilation Time : ', time.time() - start)
    return model

#-------------------------------------------------Plot
import matplotlib.pyplot as plt


'''plot results'''
def plot_predictions(result_mean, prediction_steps, predicted, global_start_time):
    try:
        test_hours_to_plot = len(predicted)
        t0 = 20  # time to start plot of predictions
        skip = 15  # skip prediction plots by specified minutes
        print('Plotting predictions...')
        plt.figure('Power Consumptions')
        # plot predicted value of t+prediction_steps as series
        # print(len(predicted),'------------------------------------')
        # val_to_be = predicted[:test_hours_to_plot , prediction_steps - 1]
        # print(len(val_to_be))

        val_sender_ubi(predicted,result_mean)
    #     plt.plot(predicted[:test_hours_to_plot , prediction_steps - 1] + result_mean,
    #              label='t+{0} prediction series'.format(prediction_steps))
    #
    #     plt.legend(loc='lower left')
    #     plt.ylabel('Actual Power in kilowatt')
    #     plt.xlabel('Time in minutes')
    #     plt.title('Predictions for first {0} minutes in test set'.format(test_hours_to_plot))
    #     plt.show()
    except Exception as e:
        print(str(e))
    # print('Duration of training (s) : ', time.time() - global_start_time)

    return None
#-------------------------------------------------Running The Lstm
def run_lstm(model, sequence_length, prediction_steps):
    global_start_time = time.time()
    ratio_of_data = 1  # ratio of data to use from 2+ million data points
    path_to_dataset = 'data/household_power_consumption.txt'

    x_test, result_mean = load_data(path_to_dataset, sequence_length, prediction_steps, ratio_of_data)
    'dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd'
    print('Data Loaded. Compiling...')
    pri_val = get_val_ubi(result_mean)
    pri_val = np.array(pri_val)
    print(pri_val)
    pri_val = pri_val.reshape(1,10)
    predicted = model.predict(pri_val)
    # print(predicted,'-------------------------89')
    print('Loading model...')
    plot_predictions(result_mean, prediction_steps, predicted, global_start_time)

    return None

# --------------------------- Run
if __name__ == '__main__':
    loading_model = True
    if loading_model:
        model = load_model('LSTM_power_consumption_model.h5')
    else:
        model = None
    sequence_length = 10  # number of past minutes of data for model to consider
    prediction_steps = 5  # number of future minutes of data for model to predict
    run_lstm(model, sequence_length, prediction_steps)


