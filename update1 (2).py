import pandas as pd
import numpy as np
import os
import webbrowser
import yfinance as yf
yf.pdr_override()
import tkinter
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from imutils import paths
from scipy.special import expit
import pickle
import random
import time
from plotly.offline import plot
import datetime
from datetime import date
from pandas_datareader import data as pdr
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pandas import Series, DataFrame
from sklearn.preprocessing import StandardScaler
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import style
from matplotlib import cm as cm
import plotly.express as px
import matplotlib.dates as mdates
import math
import numpy as np
from sklearn.metrics import accuracy_score
import seaborn as sns
from datetime import date, timedelta, datetime
from sklearn.datasets import fetch_openml
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, LeakyReLU
from keras.regularizers import l2
from keras.layers import LSTM, Bidirectional, GRU
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.optimizers import Adam
from sklearn.metrics import fbeta_score
from IPython.display import HTML
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import log_loss
import itertools
from tensorflow.keras.layers import BatchNormalization
import tensorflow.keras.backend as K
from statsmodels.tsa.stattools import kpss
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from tensorflow.keras.models import Model
from keras.utils.np_utils import to_categorical
from keras.initializers import he_normal
from tensorflow.keras.utils import plot_model
import tensorflow as tf
import seaborn as sns
sns.set_style('white', {'axes.spines.right': False, 'axes.spines.top': False})

"""# Set time & stock name"""

end_date = date.today().strftime("%Y-%m-%d")
    start_date = '2019-01-01'
    stockname = 'spy'
    symbol = 'spy'

"""# Features"""

FEATURES = ['High', 'Low', 'Open', 'Close', 'Adj Close', 'Volume']
    print('FEATURE LIST')
    print([f for f in FEATURES])

"""# DATAREADER"""

# IMPORT PANDAS_DATAREADER AS WEBREADER
    df = pdr.get_data_yahoo(symbol, start=start_date, end=end_date)
    df.head()
    df_plot = df.copy()
    # INDEXING BATCHES
    train_df = df.sort_values(by=['Date']).copy()
    print(train_df)

# Plot line charts
    df_plot = df.copy()

    ncols = 2
    nrows = int(round(df_plot.shape[1] / ncols, 0))

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=(14, 7))
    for i, ax in enumerate(fig.axes):
            sns.lineplot(data = df_plot.iloc[:, i], ax=ax)
            ax.tick_params(axis="x", rotation=30, labelsize=10, length=0)
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.tight_layout()
    plt.show()

# Plot the closing prices
    plt.figure(figsize=(10, 6))
    plt.plot(df['Close'])
    plt.title('Stock Data - Closing Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()

"""# DATAFRAME"""

data = pd.DataFrame(train_df)
    data_filtered = data[FEATURES]

    data_filtered_ext = data_filtered.copy()
    data_filtered_ext['Prediction'] = data_filtered_ext['Close']
    # PRINT THE TAIL OF THE DATAFRAME

    data_filtered_ext.tail()

"""# Correlation matrix for 6 features"""

plt.figure(figsize=(10,6))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True)
    plt.title('Correlation Matrix of Stock')
    plt.show()

"""correlation matrix is symmetrical, half of the correlation coefficients shown in the matrix are redundant and unnecessary. Thus, sometimes only half of the correlation matrix will be displayed"""

plt.figure(figsize=(10,6))
corr_matrix = df.corr()
mask = np.zeros_like(corr_matrix, dtype=bool)
mask[np.triu_indices_from(mask)] = True
corr_matrix[mask] = np.nan
sns.heatmap(corr_matrix, annot=True, cmap='Blues', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Stock')
plt.show()

"""# Using original values"""

# Load the data
    df1 = data_filtered_ext['Close']

    # Plot the original time series
    plt.figure(figsize=(15, 8))
    plt.plot(df1, label='Original')
    plt.title('Time Series - Original data')
    plt.legend()
    plt.show()

"""# using difference method on Time Series data"""

# Take the first difference
    df1_diff = df1.diff().dropna()

    # Plot the differenced time series
    plt.figure(figsize=(15, 8))
    plt.plot(df1_diff, label='Differenced')
    plt.title('Time Series - Differenced')
    plt.legend()
    plt.show()

"""# using Log method on Time Series data"""

# Take the log of the time series
    df1_log = np.log(df1)

    # Plot the logged time series
    plt.figure(figsize=(15, 8))
    plt.plot(df1_log, label='Logged')
    plt.title('Time Series - Logged')
    plt.legend()
    plt.show()

"""# using Square Rooted method on Time Series data"""

# Take the square root of the time series
    df1_sqrt = np.sqrt(df1)

    # Plot the square rooted time series
    plt.figure(figsize=(15, 8))
    plt.plot(df1_sqrt, label='Square Rooted')
    plt.title('Time Series - Square Rooted')
    plt.legend()
    plt.show()

adft2=adfuller(df['Close'].dropna(),autolag='AIC')
    output2=pd.Series(adft2[0:4],index=['Test Statistics', 'p-value','No. of lags used','Number of observations used'])
    for key,values in adft2[4].items():
        output2['critical value(%s)'%key]=values
    print(output2)

"""# Fill missing values"""

# Forward fill missing values
    df.fillna(method='ffill', inplace=True)

"""# Finding non-stationary / stationary by using original values"""

# Calculate mean and autocorrelation
    mean = df['Close'].mean()
    autocorrelation = df['Close'].autocorr()

    print('Mean:', mean)
    print('Autocorrelation:', autocorrelation)

from statsmodels.tsa.stattools import adfuller

    # Perform the ADF test
    result = adfuller(df['Close'])

    # Extract the p-value from the test result
    p_value = result[1]

    if p_value > 0.05:
        print('The series is non-stationary.')
    else:
        print('The series is stationary.')

"""# ACF & PACF befour data stationary"""

# Plot ACF
    plot_acf(df['Close'], lags=20)
    plt.title('Autocorrelation Function (ACF)')
    plt.xlabel('Lag')
    plt.ylabel('ACF')
    plt.show()

# Plot PACF using the unadjusted Yule-Walker method
    plot_pacf(df['Close'], lags=20, method='ywm')
    plt.title('Partial Autocorrelation Function (PACF)')
    plt.xlabel('Lag')
    plt.ylabel('PACF')
    plt.show()

"""# Rolmean befour data stationary"""

rolmean = data_filtered_ext['Close'].rolling(window=12).mean()
    rolstd = data_filtered_ext['Close'].rolling(window=12).std()
    plt.figure(figsize=(16,8))
    plt.plot(data_filtered_ext['Close'], color='blue', label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

"""# Bollinger bands value befour data stationary"""

# Calculate Bollinger Bands
    period = 20
    std_dev = 2
    df['SMA'] = df['Close'].rolling(window=period).mean()
    df['STD'] = df['Close'].rolling(window=period).std()
    df['Upper'] = df['SMA'] + (std_dev * df['STD'])
    df['Lower'] = df['SMA'] - (std_dev * df['STD'])

plt.figure(figsize=(16,8))
    # Plot the stock prices
    plt.plot(df['Close'], color='grey')
    # Plot the Bollinger Bands
    plt.plot(df['Upper'], label='Upper Band', color='green')
    plt.plot(df['Lower'], label='Lower Band', color='red')
    plt.plot(df['SMA'], label='SMA', color='blue')

    # Add a legend and title
    plt.legend()
    plt.title('Bollinger Bands')

    # Show the plot
    plt.show()

"""# Auto-correlation befour data stationary"""

# Calculate auto-correlation using pandas autocorr function
    auto_corr = df['Close'].autocorr()

    # Print the auto-correlation value
    print("Auto-correlation:", auto_corr)

    # Plot auto-correlation
    plt.figure(figsize=(12, 6))
    pd.plotting.autocorrelation_plot(df['Close'])
    plt.title('Auto-correlation of Stock')
    plt.show()

"""# Correlation matrix for 6 features, SAM, STD, UPPER, LOWER"""

plt.figure(figsize=(9,6))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True)
    plt.title('Correlation Matrix of Stock')
    plt.show()

"""# Using differencing to make the series stationary"""

# Perform differencing to make the series stationary
    df['Close_diff'] = df['Close'].diff()
    df.dropna(inplace=True)  # Drop the first NaN value

    # Calculate mean and autocorrelation of the differenced series
    mean_diff = df['Close_diff'].mean()
    autocorrelation_diff = df['Close_diff'].autocorr()

    print('Mean (After Differencing):', mean_diff)
    print('Autocorrelation (After Differencing):', autocorrelation_diff)

"""# Results weather data is stationary or non-stationary"""

from statsmodels.tsa.stattools import adfuller

    # Perform the ADF test
    result = adfuller(df['Close_diff'])

    # Extract the p-value from the test result
    p_value = result[1]

    if p_value > 0.05:
        print('The series is non-stationary.')
    else:
        print('The series is stationary.')

"""# ACF & PACF graphs after data stationary"""

import matplotlib.pyplot as plt
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    # Plot ACF
    plot_acf(df['Close_diff'], lags=20)
    plt.xlabel('Lags')
    plt.ylabel('Autocorrelation')
    plt.title('Autocorrelation Function (ACF)')
    plt.show()

# Plot PACF using the unadjusted Yule-Walker method
    plot_pacf(df['Close_diff'], lags=20, method='ywm')
    plt.xlabel('Lags')
    plt.ylabel('Partial Autocorrelation')
    plt.title('Partial Autocorrelation Function (PACF)')
    plt.show()

"""# Rolmean value after data stationary"""

rolmean = df['Close_diff'].rolling(window=12).mean()
    rolstd = df['Close_diff'].rolling(window=12).std()
    plt.figure(figsize=(16,8))
    plt.plot(df['Close_diff'], color='blue', label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation - After data stationary ')
    plt.show()

"""# Bollinger bands value after data stationary"""

# Calculate Bollinger Bands
    period = 20
    std_dev = 2
    df['SMA'] = df['Close_diff'].rolling(window=period).mean()
    df['STD'] = df['Close_diff'].rolling(window=period).std()
    df['Upper'] = df['SMA'] + (std_dev * df['STD'])
    df['Lower'] = df['SMA'] - (std_dev * df['STD'])

plt.figure(figsize=(16,8))
    # Plot the stock prices
    plt.plot(df['Close_diff'], color='grey')
    # Plot the Bollinger Bands
    plt.plot(df['Upper'], label='Upper Band', color='green')
    plt.plot(df['Lower'], label='Lower Band', color='red')
    plt.plot(df['SMA'], label='SMA', color='blue')

    # Add a legend and title
    plt.legend()
    plt.title('Bollinger Bands')

    # Show the plot
    plt.show()

"""# Auto-correlation value after data stationary"""

# Calculate auto-correlation using pandas autocorr function
    auto_corr = df['Close_diff'].autocorr()

    # Print the auto-correlation value
    print("Auto-correlation:", auto_corr)

    # Plot auto-correlation
    plt.figure(figsize=(12, 6))
    pd.plotting.autocorrelation_plot(df['Close_diff'])
    plt.title('Auto-correlation of Stock')
    plt.show()

"""# GETING DATA IN ROWS.

# CONVERT NUMPY VALUES TO DATA.
"""

# GET THE NUMBER OF ROWS IN THE DATA
    nrows = data_filtered.shape[0]
    # CONVERT THE DATA TO NUMPY VALUES
    np_data_unscaled = np.array(data_filtered)
    np_data = np.reshape(np_data_unscaled, (nrows, -1))
    print(np_data.shape)

print(data_filtered_ext.shape)

"""# TRANSFORM THE DATA BY SCALING FOR EACH FEATURE"""

# TRANSFORM THE DATA BY SCALING EACH FEATURE TO A RANGE BETWEEN 0 AND 1
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    np_data_scaled = scaler.fit_transform(np_data_unscaled)

    # CREATING A SEPARATE SCALER THAT WORKS ON A SINGLE COLUMN FOR SCALING PREDICTIONS

    scaler_pred = preprocessing.MinMaxScaler(feature_range=(0, 1))
    df_Close = pd.DataFrame(data_filtered_ext['Close'])
    np_Close_scaled = scaler_pred.fit_transform(df_Close)

"""# Set the sequence length to make a single prediction"""

# Set the sequence length - this is the timeframe used to make a single prediction
    sequence_length = 50

    # Prediction Index
    index_Close = data.columns.get_loc("Close")

    # Split the training data into train and train data sets
    # As a first step, we get the number of rows to train the model on 80% of the data
    train_data_len = math.ceil(np_data_scaled.shape[0] * 0.8)

    # Create the training and test data
    train_data = np_data_scaled[0:train_data_len, :]
    test_data = np_data_scaled[train_data_len - sequence_length:, :]

"""# create N samples, sequence_length time steps per sample"""

# The RNN needs data with the format of [samples, time steps, features]
    # Here, we create N samples, sequence_length time steps per sample, and 6 features
    def partition_dataset(sequence_length, data):
        x, y = [], []
        data_len = data.shape[0]
        for i in range(sequence_length, data_len):
            x.append(data[i-sequence_length:i,:]) #contains sequence_length values 0-sequence_length * columsn
            y.append(data[i, index_Close]) #contains the prediction values for validation (3rd column = Close),  for single-step prediction

        # Convert the x and y to numpy arrays
        x = np.array(x)
        y = np.array(y)
        return x, y

"""# Generate training data and test data"""

# Generate training data and test data
    x_train, y_train = partition_dataset(sequence_length, train_data)
    x_test, y_test = partition_dataset(sequence_length, test_data)

    # Print the shapes: the result is: (rows, training_sequence, features) (prediction value, )
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)

    # Validate that the prediction value and the input match up
    # The last close price of the second input sample should equal the first prediction value
    print(x_test[1][sequence_length-1][index_Close])
    print(y_test[0])

"""# Only use one model at a time

# Method - 1
"""

# Define the model architecture
    model = Sequential()

    # Add a LSTM layer
    n_neurons = x_train.shape[1] * x_train.shape[2]

    model.add(LSTM(n_neurons * 2, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dropout(0.2))

    model.add(LSTM(n_neurons * 2, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(n_neurons * 2, return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(25, activation='ReLU'))
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01)))

    model.summary()

"""# method - 2"""

# Define the model architecture
    model2 = Sequential()

    # Add a bidirectional LSTM layer
    n_neurons = x_train.shape[1] * x_train.shape[2]

    model2.add(Bidirectional(LSTM(n_neurons * 2, return_sequences=True), input_shape=(x_train.shape[1], x_train.shape[2])))
    model2.add(Dropout(0.2))

    model2.add(Bidirectional(LSTM(n_neurons * 2, return_sequences=True)))
    model2.add(Dropout(0.2))

    model2.add(Bidirectional(LSTM(n_neurons * 2, return_sequences=False)))
    model2.add(Dropout(0.2))

    model2.add(Dense(25, activation='ReLU'))
    model2.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01)))

    model2.summary()

"""# Method - 3 using attention layer"""

from keras.layers import Dense, LSTM, Dropout, Bidirectional, Activation, Layer
    from keras import Input, Model
    from keras.regularizers import l2
    import keras.backend as K

    # Define the custom attention layer
    class AttentionLayer(Layer):
        def __init__(self):
            super(AttentionLayer, self).__init__()

        def build(self, input_shape):
            self.W = self.add_weight(shape=(input_shape[-1], 1),
                                    initializer='glorot_uniform',
                                    trainable=True)
            super(AttentionLayer, self).build(input_shape)

        def call(self, inputs):
            attention_weights = K.dot(inputs, self.W)
            attention_weights = K.squeeze(attention_weights, axis=-1)
            attention_weights = K.softmax(attention_weights, axis=-1)
            attention_weights = K.expand_dims(attention_weights, axis=-1)
            attention_output = inputs * attention_weights
            return attention_output

        def compute_output_shape(self, input_shape):
            return input_shape

    # Define the model architecture
    inputs = Input(shape=(x_train.shape[1], x_train.shape[2]))
    n_neurons = x_train.shape[1] * x_train.shape[2]

    lstm_output = Bidirectional(LSTM(n_neurons * 2, return_sequences=True))(inputs)
    attention_output = AttentionLayer()(lstm_output)
    attention_output = LSTM(n_neurons * 2, return_sequences=False)(attention_output)
    attention_output = Dropout(0.2)(attention_output)

    output = Dense(25, activation='relu')(attention_output)
    output = Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01))(output)

    model3 = Model(inputs=inputs, outputs=output)
    model3.summary()

"""# Method - 3(1) add more layers using attention layers"""

from keras.layers import Dense, LSTM, Dropout, Bidirectional, Activation, Layer
    from keras import Input, Model
    from keras.regularizers import l2
    import keras.backend as K

    # Define the custom attention layer
    class AttentionLayer(Layer):
        def __init__(self):
            super(AttentionLayer, self).__init__()

        def build(self, input_shape):
            self.W = self.add_weight(shape=(input_shape[-1], 1),
                                    initializer='glorot_uniform',
                                    trainable=True)
            super(AttentionLayer, self).build(input_shape)

        def call(self, inputs):
            attention_weights = K.dot(inputs, self.W)
            attention_weights = K.squeeze(attention_weights, axis=-1)
            attention_weights = K.softmax(attention_weights, axis=-1)
            attention_weights = K.expand_dims(attention_weights, axis=-1)
            attention_output = inputs * attention_weights
            return attention_output

        def compute_output_shape(self, input_shape):
            return input_shape

    # Define the model architecture
    inputs = Input(shape=(x_train.shape[1], x_train.shape[2]))
    n_neurons = x_train.shape[1] * x_train.shape[2]

    lstm_output = Bidirectional(LSTM(n_neurons * 2, return_sequences=True))(inputs)
    lstm_output = Dropout(0.2)(lstm_output)

    lstm_output = Bidirectional(LSTM(n_neurons * 2, return_sequences=True))(lstm_output)
    lstm_output = Dropout(0.2)(lstm_output)

    lstm_output = Bidirectional(LSTM(n_neurons * 2, return_sequences=False))(lstm_output)
    lstm_output = Dropout(0.2)(lstm_output)

    attention_output = AttentionLayer()(lstm_output)

    output = Dense(25, activation='relu')(attention_output)
    output = Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01))(output)

    model3a = Model(inputs=inputs, outputs=output)
    model3a.summary()

"""# Method - 4"""

# Define the model architecture
    model4 = Sequential()

    # Add a bidirectional GRU layer
    n_neurons = x_train.shape[1] * x_train.shape[2]

    model4.add(Bidirectional(GRU(n_neurons * 2, return_sequences=True), input_shape=(x_train.shape[1], x_train.shape[2])))
    model4.add(Dropout(0.2))

    model4.add(Bidirectional(GRU(n_neurons * 2, return_sequences=True)))
    model4.add(Dropout(0.2))

    model4.add(Bidirectional(GRU(n_neurons * 2, return_sequences=False)))
    model4.add(Dropout(0.2))

    model4.add(Dense(25, activation='ReLU'))
    model4.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01)))

    model4.summary()

"""# Building image of model structure"""

tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

tf.keras.utils.plot_model(model2, to_file='model2.png', show_shapes=True, show_layer_names=True)

tf.keras.utils.plot_model(model3, to_file='model3.png', show_shapes=True, show_layer_names=True)

tf.keras.utils.plot_model(model3a, to_file='model3a.png', show_shapes=True, show_layer_names=True)

tf.keras.utils.plot_model(model4, to_file='model4.png', show_shapes=True, show_layer_names=True)

""" # COMPILE THE MODEL"""

# COMPILE THE MODEL
    model.compile(loss='MSE', optimizer="adam")
    model2.compile(loss='MSE', optimizer="adam")
    model3.compile(loss='MSE', optimizer="adam")
    model3a.compile(loss='MSE', optimizer="adam")
    model4.compile(loss='MSE', optimizer="adam")

"""# TRAINING THE MODEL"""

# TRAINING THE MODEL
    epochs = 100
    batch_size = 16
    early_stop = EarlyStopping(monitor='loss', patience=4, verbose=1)

    # apply data augmentation to increase the size and diversity of the data
    x_train_augmented = np.concatenate([x_train, x_train * 0.9, x_train * 1.1]) # multiply the data by 0.9 and 1.1 to create new samples
    y_train_augmented = np.concatenate([y_train, y_train * 0.9, y_train * 1.1]) # multiply the labels by 0.9 and 1.1 to create new labels

    # add some noise to the input data to make the model more robust
    noise_factor = 0.01
    x_train_noisy = x_train_augmented + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train_augmented.shape)
    x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

"""# model-1"""

start_time = time.time() # get the start time
    # train the model with early stopping and validation data
    history_model = model.fit(x_train_noisy, y_train_augmented,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_test_noisy, y_test)
                        ) # train the model
    end_time = time.time() # get the end time

elapsed_time = end_time - start_time # calculate the elapsed time
    print(f'The model took {elapsed_time} seconds to execute.') # print the result

"""# model-2"""

start_time = time.time() # get the start time
    # train the model with early stopping and validation data
    history_model2 = model2.fit(x_train_noisy, y_train_augmented,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_test_noisy, y_test)
                        ) # train the model
    end_time = time.time() # get the end time
    elapsed_time = end_time - start_time # calculate the elapsed time
    print(f'The model2 took {elapsed_time} seconds to execute.') # print the result

"""# model-3"""

start_time = time.time() # get the start time
    # train the model with early stopping and validation data
    history_model3 = model3.fit(x_train_noisy, y_train_augmented,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_test_noisy, y_test)
                        ) # train the model
    end_time = time.time() # get the end time
    elapsed_time = end_time - start_time # calculate the elapsed time
    print(f'The model3 took {elapsed_time} seconds to execute.') # print the result

"""# model-3a"""

start_time = time.time() # get the start time
    # train the model with early stopping and validation data
    history_model3a = model3a.fit(x_train_noisy, y_train_augmented,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_test_noisy, y_test)
                        ) # train the model
    end_time = time.time() # get the end time
    elapsed_time = end_time - start_time # calculate the elapsed time
    print(f'The model3a took {elapsed_time} seconds to execute.') # print the result

"""# model - 4"""

start_time = time.time() # get the start time
    # train the model with early stopping and validation data
    history_model4 = model4.fit(x_train_noisy, y_train_augmented,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_test_noisy, y_test)
                        ) # train the model
    end_time = time.time() # get the end time
    elapsed_time = end_time - start_time # calculate the elapsed time
    print(f'The model4 took {elapsed_time} seconds to execute.') # print the result

"""# GET THE PREDICTED VALUES Model"""

# GET THE PREDICTED VALUES Model
    y_pred_scaled = model.predict(x_test_noisy)
    y_pred_scaled2 = model2.predict(x_test_noisy)
    y_pred_scaled3 = model3.predict(x_test_noisy)
    y_pred_scaled3a = model3a.predict(x_test_noisy)
    y_pred_scaled4 = model4.predict(x_test_noisy)

"""# Evaluate model"""

# Evaluate model
    loss = model.evaluate(x_test_noisy, y_test)

"""# Plot training & validation loss values

# model-1
"""

# Plot training & validation loss values
    fig, ax = plt.subplots(figsize=(16, 8), sharex=True)
    plt.plot(history_model.history["val_loss"])
    plt.plot(history_model.history["loss"])
    plt.title("Model loss",fontsize=14)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    ax.xaxis.set_major_locator(plt.MaxNLocator(epochs))
    plt.legend(["Val", "Loss"], loc="upper left")
    plt.grid()
    plt.show()

train_loss = history_model.history['loss']
    val_loss = history_model.history['val_loss']
    # Plot the loss and RMSE
    RMSE = np.sqrt(val_loss)
    plt.figure(figsize=(16, 8))
    plt.plot(RMSE, label='Validation RMSE')
    plt.legend(loc='upper right')
    plt.title('Validation RMSE')
    plt.show()

"""# model - 2"""

# Evaluate model
    loss = model2.evaluate(x_test_noisy, y_test)

# Plot training & validation loss values
    fig, ax = plt.subplots(figsize=(16, 8), sharex=True)
    plt.plot(history_model2.history["val_loss"])
    plt.plot(history_model2.history["loss"])
    plt.title("Model loss",fontsize=14)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    ax.xaxis.set_major_locator(plt.MaxNLocator(epochs))
    plt.legend(["Val", "Loss"], loc="upper left")
    plt.grid()
    plt.show()

train_loss = history_model2.history['loss']
    val_loss = history_model2.history['val_loss']
    # Plot the loss and RMSE
    RMSE = np.sqrt(val_loss)
    plt.figure(figsize=(16, 8))
    plt.plot(RMSE, label='Validation RMSE')
    plt.legend(loc='upper right')
    plt.title('Validation RMSE')
    plt.show()

"""# model -3"""

# Evaluate model
    loss = model2.evaluate(x_test_noisy, y_test)

# Plot training & validation loss values
    fig, ax = plt.subplots(figsize=(16, 8), sharex=True)
    plt.plot(history_model3.history["val_loss"])
    plt.plot(history_model3.history["loss"])
    plt.title("Model loss",fontsize=14)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    ax.xaxis.set_major_locator(plt.MaxNLocator(epochs))
    plt.legend(["Val", "Loss"], loc="upper left")
    plt.grid()
    plt.show()

train_loss = history_model3.history['loss']
    val_loss = history_model3.history['val_loss']
    # Plot the loss and RMSE
    RMSE = np.sqrt(val_loss)
    plt.figure(figsize=(16, 8))
    plt.plot(RMSE, label='Validation RMSE')
    plt.legend(loc='upper right')
    plt.title('Validation RMSE')
    plt.show()

"""# model - 3a"""

# Evaluate model
    loss = model3a.evaluate(x_test_noisy, y_test)

# Plot training & validation loss values
    fig, ax = plt.subplots(figsize=(16, 8), sharex=True)
    plt.plot(history_model3a.history["val_loss"])
    plt.plot(history_model3a.history["loss"])
    plt.title("Model loss",fontsize=14)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    ax.xaxis.set_major_locator(plt.MaxNLocator(epochs))
    plt.legend(["Val", "Loss"], loc="upper left")
    plt.grid()
    plt.show()

train_loss = history_model3a.history['loss']
    val_loss = history_model3a.history['val_loss']
    # Plot the loss and RMSE
    RMSE = np.sqrt(val_loss)
    plt.figure(figsize=(16, 8))
    plt.plot(RMSE, label='Validation RMSE')
    plt.legend(loc='upper right')
    plt.title('Validation RMSE')
    plt.show()

"""# model - 4"""

# Evaluate model
    loss = model4.evaluate(x_test_noisy, y_test)

# Plot training & validation loss values
    fig, ax = plt.subplots(figsize=(16, 8), sharex=True)
    plt.plot(history_model4.history["val_loss"])
    plt.plot(history_model4.history["loss"])
    plt.title("Model loss",fontsize=14)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    ax.xaxis.set_major_locator(plt.MaxNLocator(epochs))
    plt.legend(["Val", "Loss"], loc="upper left")
    plt.grid()
    plt.show()

train_loss = history_model4.history['loss']
    val_loss = history_model4.history['val_loss']
    # Plot the loss and RMSE
    RMSE = np.sqrt(val_loss)
    plt.figure(figsize=(16, 8))
    plt.plot(RMSE, label='Validation RMSE')
    plt.legend(loc='upper right')
    plt.title('Validation RMSE')
    plt.show()

"""# Reshaping Data"""

y_pred = scaler_pred.inverse_transform(y_pred_scaled)
    y_test_unscaled = scaler_pred.inverse_transform(y_test.reshape(-1, 1))

y_pred2 = scaler_pred.inverse_transform(y_pred_scaled2)
    y_test_unscaled2 = scaler_pred.inverse_transform(y_test.reshape(-1, 1))

y_pred3 = scaler_pred.inverse_transform(y_pred_scaled3)
    y_test_unscaled3 = scaler_pred.inverse_transform(y_test.reshape(-1, 1))

y_pred3a = scaler_pred.inverse_transform(y_pred_scaled3a)
    y_test_unscaled3a = scaler_pred.inverse_transform(y_test.reshape(-1, 1))

y_pred4 = scaler_pred.inverse_transform(y_pred_scaled4)
    y_test_unscaled4 = scaler_pred.inverse_transform(y_test.reshape(-1, 1))

"""# Calculating error

# model-1
"""

# Mean Absolute Error (MAE)
    MAE_model = np.mean(abs(y_pred - y_test_unscaled))
    print('Mean Absolute Error (MAE): ' + str(np.round(MAE_model, 2)))

    # Median Absolute Error (MedAE)
    MEDAE_model = np.median(abs(y_pred - y_test_unscaled))
    print('Median Absolute Error (MEDAE): ' + str(np.round(MEDAE_model, 2)))

    # Mean Squared Error (MSE)
    MSE_model = np.square(np.subtract(y_pred, y_test_unscaled)).mean()
    print('Mean Squared Error (MSE): ' + str(np.round(MSE_model, 2)))

    # Root Mean Squarred Error (RMSE)
    RMSE_model = np.sqrt(np.mean(np.square(y_pred - y_test_unscaled)))
    print('Root Mean Squared Error (RMSE): ' + str(np.round(RMSE_model, 2)))

    # Mean Absolute Percentage Error (MAPE)
    MAPE_model = np.mean((np.abs(np.subtract(y_test_unscaled, y_pred)/ y_test_unscaled))) * 100
    print('Mean Absolute Percentage Error (MAPE): ' + str(np.round(MAPE_model, 2)) + ' %')

    # Median Absolute Percentage Error (MDAPE)
    MDAPE_model = np.median((np.abs(np.subtract(y_test_unscaled, y_pred)/ y_test_unscaled))) * 100
    print('Median Absolute Percentage Error (MDAPE): ' + str(np.round(MDAPE_model, 2)) + ' %')

"""# model - 2"""

# Mean Absolute Error (MAE)
    MAE2 = np.mean(abs(y_pred2 - y_test_unscaled2))
    print('Mean Absolute Error (MAE): ' + str(np.round(MAE2, 2)))

    # Median Absolute Error (MedAE)
    MEDAE2 = np.median(abs(y_pred2 - y_test_unscaled2))
    print('Median Absolute Error (MEDAE): ' + str(np.round(MEDAE2, 2)))

    # Mean Squared Error (MSE)
    MSE2 = np.square(np.subtract(y_pred2, y_test_unscaled2)).mean()
    print('Mean Squared Error (MSE): ' + str(np.round(MSE2, 2)))

    # Root Mean Squarred Error (RMSE)
    RMSE2 = np.sqrt(np.mean(np.square(y_pred2 - y_test_unscaled2)))
    print('Root Mean Squared Error (RMSE): ' + str(np.round(RMSE2, 2)))

    # Mean Absolute Percentage Error (MAPE)
    MAPE2 = np.mean((np.abs(np.subtract(y_test_unscaled2, y_pred2)/ y_test_unscaled2))) * 100
    print('Mean Absolute Percentage Error (MAPE): ' + str(np.round(MAPE2, 2)) + ' %')

    # Median Absolute Percentage Error (MDAPE)
    MDAPE2 = np.median((np.abs(np.subtract(y_test_unscaled2, y_pred2)/ y_test_unscaled2))) * 100
    print('Median Absolute Percentage Error (MDAPE): ' + str(np.round(MDAPE2, 2)) + ' %')

"""# model - 3"""

# Mean Absolute Error (MAE)
    MAE3 = np.mean(abs(y_pred3 - y_test_unscaled3))
    print('Mean Absolute Error (MAE): ' + str(np.round(MAE3, 2)))

    # Median Absolute Error (MedAE)
    MEDAE3 = np.median(abs(y_pred3 - y_test_unscaled3))
    print('Median Absolute Error (MEDAE): ' + str(np.round(MEDAE3, 2)))

    # Mean Squared Error (MSE)
    MSE3 = np.square(np.subtract(y_pred3, y_test_unscaled3)).mean()
    print('Mean Squared Error (MSE): ' + str(np.round(MSE3, 2)))

    # Root Mean Squarred Error (RMSE)
    RMSE3 = np.sqrt(np.mean(np.square(y_pred3 - y_test_unscaled3)))
    print('Root Mean Squared Error (RMSE): ' + str(np.round(RMSE3, 2)))

    # Mean Absolute Percentage Error (MAPE)
    MAPE3 = np.mean((np.abs(np.subtract(y_test_unscaled3, y_pred3)/ y_test_unscaled3))) * 100
    print('Mean Absolute Percentage Error (MAPE): ' + str(np.round(MAPE3, 2)) + ' %')

    # Median Absolute Percentage Error (MDAPE)
    MDAPE3 = np.median((np.abs(np.subtract(y_test_unscaled3, y_pred3)/ y_test_unscaled3))) * 100
    print('Median Absolute Percentage Error (MDAPE): ' + str(np.round(MDAPE3, 2)) + ' %')

"""# model - 3a"""

# Mean Absolute Error (MAE)
    MAE3a = np.mean(abs(y_pred3a - y_test_unscaled3a))
    print('Mean Absolute Error (MAE): ' + str(np.round(MAE3a, 2)))

    # Median Absolute Error (MedAE)
    MEDAE3a = np.median(abs(y_pred3a - y_test_unscaled3a))
    print('Median Absolute Error (MEDAE): ' + str(np.round(MEDAE3a, 2)))

    # Mean Squared Error (MSE)
    MSE3a = np.square(np.subtract(y_pred3a, y_test_unscaled3a)).mean()
    print('Mean Squared Error (MSE): ' + str(np.round(MSE3a, 2)))

    # Root Mean Squarred Error (RMSE)
    RMSE3a = np.sqrt(np.mean(np.square(y_pred3a - y_test_unscaled3a)))
    print('Root Mean Squared Error (RMSE): ' + str(np.round(RMSE3a, 2)))

    # Mean Absolute Percentage Error (MAPE)
    MAPE3a = np.mean((np.abs(np.subtract(y_test_unscaled3a, y_pred3a)/ y_test_unscaled3a))) * 100
    print('Mean Absolute Percentage Error (MAPE): ' + str(np.round(MAPE3a, 2)) + ' %')

    # Median Absolute Percentage Error (MDAPE)
    MDAPE3a = np.median((np.abs(np.subtract(y_test_unscaled3a, y_pred3a)/ y_test_unscaled3a))) * 100
    print('Median Absolute Percentage Error (MDAPE): ' + str(np.round(MDAPE3a, 2)) + ' %')

"""# model - 4"""

# Mean Absolute Error (MAE)
    MAE4 = np.mean(abs(y_pred4 - y_test_unscaled4))
    print('Mean Absolute Error (MAE): ' + str(np.round(MAE4, 2)))

    # Median Absolute Error (MedAE)
    MEDAE4 = np.median(abs(y_pred4 - y_test_unscaled4))
    print('Median Absolute Error (MEDAE): ' + str(np.round(MEDAE4, 2)))

    # Mean Squared Error (MSE)
    MSE4 = np.square(np.subtract(y_pred4, y_test_unscaled4)).mean()
    print('Mean Squared Error (MSE): ' + str(np.round(MSE4, 2)))

    # Root Mean Squarred Error (RMSE)
    RMSE4 = np.sqrt(np.mean(np.square(y_pred4 - y_test_unscaled4)))
    print('Root Mean Squared Error (RMSE): ' + str(np.round(RMSE4, 2)))

    # Mean Absolute Percentage Error (MAPE)
    MAPE4 = np.mean((np.abs(np.subtract(y_test_unscaled4, y_pred4)/ y_test_unscaled4))) * 100
    print('Mean Absolute Percentage Error (MAPE): ' + str(np.round(MAPE4, 2)) + ' %')

    # Median Absolute Percentage Error (MDAPE)
    MDAPE4 = np.median((np.abs(np.subtract(y_test_unscaled4, y_pred4)/ y_test_unscaled4))) * 100
    print('Median Absolute Percentage Error (MDAPE): ' + str(np.round(MDAPE4, 2)) + ' %')

"""# R2 score of predicted data"""

from sklearn.metrics import r2_score

"""# model -1"""

Acc = []
    print("R2 score: {0}".format(r2_score(y_test, y_pred_scaled)))
    Acc.append(r2_score(y_test, y_pred_scaled))

"""# model - 2"""

Acc_model2 = []
    print("R2 score: {0}".format(r2_score(y_test, y_pred_scaled2)))
    Acc_model2.append(r2_score(y_test, y_pred_scaled2))

"""# model -3"""

Acc_model3 = []
    print("R2 score: {0}".format(r2_score(y_test, y_pred_scaled3)))
    Acc_model3.append(r2_score(y_test, y_pred_scaled3))

"""# model - 3a"""

Acc_model3a = []
    print("R2 score: {0}".format(r2_score(y_test, y_pred_scaled3a)))
    Acc_model3a.append(r2_score(y_test, y_pred_scaled3a))

"""# model - 4"""

Acc_model4 = []
    print("R2 score: {0}".format(r2_score(y_test, y_pred_scaled4)))
    Acc_model4.append(r2_score(y_test, y_pred_scaled4))

"""# model - 1

# round predicted data for better understanding
"""

y_pred = np.round(y_pred)

df_temp = df[-sequence_length:]
    new_df = df_temp.filter(FEATURES)

"""# create a new test for predicted data"""

N = sequence_length

    # GET THE LAST N DAY CLOSING PRICE VALUES AND SCALE THE DATA TO BE VALUES BETWEEN 0 AND 1

    last_N_days = new_df[-sequence_length:].values
    last_N_days_scaled = scaler.transform(last_N_days)

    # CREATE AN EMPTY LIST AND APPEND PAST N DAYS

    test_new = []
    test_new.append(last_N_days_scaled)

    # CONVERT THE X_TEST DATA SET TO A NUMPY ARRAY AND RESHAPE THE DATA

    pred_price_scaled = model.predict(np.array(test_new))
    pred_price_unscaled = scaler_pred.inverse_transform(pred_price_scaled.reshape(-1, 1))

"""# Print Predicted Value"""

print(pred_price_unscaled)

"""# Print Actual Value"""

price_today_model = np.round(new_df['Close'][-1], 2)
    predicted_price_model = np.round(pred_price_unscaled.ravel()[0], 2)
    change_percent = np.round(100 - (price_today_model * 100)/predicted_price_model, 2)
    print(price_today_model)

"""# Compair Both actual and predicted values

"""

plus = '+'; minus = ''
    print(f'The close price for {stockname} at {end_date} was {price_today_model}')
    print(f'The next day predicted close price is {predicted_price_model} ({plus if change_percent > 0 else minus}{change_percent}%)')

"""# Create the line plot garph"""

# Create the line plot
    test_df = pd.DataFrame({'y_test': y_test_unscaled.flatten(), 'y_pred': y_pred.flatten()})
    fig, ax1 = plt.subplots(figsize=(16, 8), sharex=True)
    sns.lineplot(data=test_df)
    ax1.tick_params(axis="x", rotation=0, labelsize=10, length=0)
    plt.title("y_pred vs y_test Truth")
    plt.legend(["y_pred", "y_test"], loc="upper left")

    # Fill between plotlines
    mpl.rc('hatch', color='k', linewidth=2)
    ax1.fill_between(test_df.index, test_df["y_test"], test_df["y_pred"],  facecolor = 'white', alpha=.9)
    plt.show()

# THE DATE FROM WHICH ON THE DATE IS DISPLAYED

    display_start_date = "2019-01-01"

    # ADD THE DIFFERENCE BETWEEN THE VALID AND PREDICTED PRICES

    train = pd.DataFrame(data_filtered_ext['Close'][:train_data_len + 1]).rename(columns={'Close': 'y_train'})
    valid = pd.DataFrame(data_filtered_ext['Close'][train_data_len:]).rename(columns={'Close': 'y_test'})
    valid.insert(1, "y_pred", y_pred, True)
    valid.insert(1, "residuals", valid["y_pred"] - valid["y_test"], True)
    df_union = pd.concat([train, valid], sort=True )



    # ZOOM IN TO A CLOSER TIMEFRAME

    df_union_zoom = df_union[df_union.index > display_start_date]


    # CREATE THE LINEPLOT

    fig, ax1 = plt.subplots(figsize=(16, 8))
    plt.title("Original Price vs Predicted Price")
    plt.ylabel(stockname, fontsize=18)
    sns.set_style('white', { 'axes.spines.right': False, 'axes.spines.top': False})
    sns.set_palette(["#090364", "#1960EF", "#EF5919"])
    sns.lineplot(data=df_union_zoom[['y_pred', 'y_train', 'y_test']], linewidth=1.0, dashes=False, ax=ax1)
    # plt.fill_between(df_union_zoom.index, df_union_zoom['y_train'],df_union_zoom['y_test'], df_union_zoom['y_pred'], alpha=0.3)


    # Create the bar plot with the differences

    df_sub = ["#2BC97A" if x > 0 else "#C92B2B" for x in df_union_zoom["residuals"].dropna()]
    ax1.bar(height=df_union_zoom['residuals'].dropna(), x=df_union_zoom['residuals'].dropna().index, width=3, label='residuals', color=df_sub)

    plt.legend()
    plt.show()

"""# model - 2"""

# Round predicted data for Model 2
    y_pred_model2 = np.round(y_pred2)

    df_temp_model2 = df[-sequence_length:]
    new_df_model2 = df_temp_model2.filter(FEATURES)


    # Create a new test for predicted data in Model 2
    N_model2 = sequence_length
    last_N_days_model2 = new_df_model2[-sequence_length:].values
    last_N_days_scaled_model2 = scaler.transform(last_N_days_model2)

    test_new_model2 = []
    test_new_model2.append(last_N_days_scaled_model2)

    pred_price_scaled_model2 = model.predict(np.array(test_new_model2))
    pred_price_unscaled_model2 = scaler_pred.inverse_transform(pred_price_scaled_model2.reshape(-1, 1))

    print(pred_price_unscaled_model2)

    price_today_model2 = np.round(new_df_model2['Close'][-1], 2)
    predicted_price_model2 = np.round(pred_price_unscaled_model2.ravel()[0], 2)
    change_percent_model2 = np.round(100 - (price_today_model2 * 100) / predicted_price_model2, 2)
    print(price_today_model2)

    plus = '+'; minus = ''
    print(f'The close price for {stockname} at {end_date} was {price_today_model2}')
    print(f'The next day predicted close price is {predicted_price_model2} ({plus if change_percent > 0 else minus}{change_percent}%)')

    # Create the line plot graph for Model 2
    test_df_model2 = pd.DataFrame({'y_test': y_test_unscaled2.flatten(), 'y_pred': y_pred_model2.flatten()})
    fig_model2, ax1_model2 = plt.subplots(figsize=(16, 8), sharex=True)
    sns.lineplot(data=test_df_model2)
    ax1_model2.tick_params(axis="x", rotation=0, labelsize=10, length=0)
    plt.title("y_pred vs y_test Truth - Model 2")
    plt.legend(["y_pred", "y_test"], loc="upper left")
    mpl.rc('hatch', color='k', linewidth=2)
    ax1_model2.fill_between(test_df_model2.index, test_df_model2["y_test"], test_df_model2["y_pred"], facecolor='white', alpha=.9)
    plt.show()

    # Create the line plot for the residuals in Model 2
    train_model2 = pd.DataFrame(data_filtered_ext['Close'][:train_data_len + 1]).rename(columns={'Close': 'y_train'})
    valid_model2 = pd.DataFrame(data_filtered_ext['Close'][train_data_len:]).rename(columns={'Close': 'y_test'})
    valid_model2.insert(1, "y_pred", y_pred_model2, True)
    valid_model2.insert(1, "residuals", valid_model2["y_pred"] - valid_model2["y_test"], True)
    df_union_model2 = pd.concat([train_model2, valid_model2], sort=True)

    df_union_zoom_model2 = df_union_model2[df_union_model2.index > display_start_date]

    fig_model2, ax1_model2 = plt.subplots(figsize=(16, 8))
    plt.title("Original Price vs Predicted Price - Model 2")
    plt.ylabel(stockname, fontsize=18)
    sns.set_style('white', {'axes.spines.right': False, 'axes.spines.top': False})
    sns.set_palette(["#090364", "#1960EF", "#EF5919"])
    sns.lineplot(data=df_union_zoom_model2[['y_pred', 'y_train', 'y_test']], linewidth=1.0, dashes=False, ax=ax1_model2)
    df_sub_model2 = ["#2BC97A" if x > 0 else "#C92B2B" for x in df_union_zoom_model2["residuals"].dropna()]
    ax1_model2.bar(height=df_union_zoom_model2['residuals'].dropna(), x=df_union_zoom_model2['residuals'].dropna().index, width=3, label='residuals', color=df_sub_model2)
    plt.legend()
    plt.show()

"""# Model - 3"""

# Round predicted data for Model 3
    y_pred_model3 = np.round(y_pred3)

    df_temp_model3 = df[-sequence_length:]
    new_df_model3 = df_temp_model3.filter(FEATURES)


    # Create a new test for predicted data in Model 3
    N_model3 = sequence_length
    last_N_days_model3 = new_df_model3[-sequence_length:].values
    last_N_days_scaled_model3 = scaler.transform(last_N_days_model3)

    test_new_model3 = []
    test_new_model3.append(last_N_days_scaled_model3)

    pred_price_scaled_model3 = model.predict(np.array(test_new_model3))
    pred_price_unscaled_model3 = scaler_pred.inverse_transform(pred_price_scaled_model3.reshape(-1, 1))

    print(pred_price_unscaled_model3)

    price_today_model3 = np.round(new_df_model3['Close'][-1], 2)
    predicted_price_model3 = np.round(pred_price_unscaled_model3.ravel()[0], 2)
    change_percent_model3 = np.round(100 - (price_today_model3 * 100) / predicted_price_model3, 2)
    print(price_today_model3)

    plus = '+'; minus = ''
    print(f'The close price for {stockname} at {end_date} was {price_today_model3}')
    print(f'The next day predicted close price is {predicted_price_model3} ({plus if change_percent_model3 > 0 else minus}{change_percent_model3}%)')

    # Create the line plot graph for Model 3
    test_df_model3 = pd.DataFrame({'y_test': y_test_unscaled3.flatten(), 'y_pred': y_pred_model3.flatten()})
    fig_model3, ax1_model3 = plt.subplots(figsize=(16, 8), sharex=True)
    sns.lineplot(data=test_df_model3)
    ax1_model3.tick_params(axis="x", rotation=0, labelsize=10, length=0)
    plt.title("y_pred vs y_test Truth - Model 3")
    plt.legend(["y_pred", "y_test"], loc="upper left")
    mpl.rc('hatch', color='k', linewidth=2)
    ax1_model3.fill_between(test_df_model3.index, test_df_model3["y_test"], test_df_model3["y_pred"], facecolor='white', alpha=.9)
    plt.show()

    # Create the line plot for the residuals in Model 3
    train_model3 = pd.DataFrame(data_filtered_ext['Close'][:train_data_len + 1]).rename(columns={'Close': 'y_train'})
    valid_model3 = pd.DataFrame(data_filtered_ext['Close'][train_data_len:]).rename(columns={'Close': 'y_test'})
    valid_model3.insert(1, "y_pred", y_pred_model3, True)
    valid_model3.insert(1, "residuals", valid_model3["y_pred"] - valid_model3["y_test"], True)
    df_union_model3 = pd.concat([train_model3, valid_model3], sort=True)

    df_union_zoom_model3 = df_union_model3[df_union_model3.index > display_start_date]

    fig_model3, ax1_model3 = plt.subplots(figsize=(16, 8))
    plt.title("Original Price vs Predicted Price - Model 3")
    plt.ylabel(stockname, fontsize=18)
    sns.set_style('white', {'axes.spines.right': False, 'axes.spines.top': False})
    sns.set_palette(["#090364", "#1960EF", "#EF5919"])
    sns.lineplot(data=df_union_zoom_model3[['y_pred', 'y_train', 'y_test']], linewidth=1.0, dashes=False, ax=ax1_model3)
    df_sub_model3 = ["#2BC97A" if x > 0 else "#C92B2B" for x in df_union_zoom_model3["residuals"].dropna()]
    ax1_model3.bar(height=df_union_zoom_model3['residuals'].dropna(), x=df_union_zoom_model3['residuals'].dropna().index, width=3, label='residuals', color=df_sub_model3)
    plt.legend()
    plt.show()

"""# Model - 3a"""

# Round predicted data for Model 3a
    y_pred_model3a = np.round(y_pred3a)

    df_temp_model3a = df[-sequence_length:]
    new_df_model3a = df_temp_model3a.filter(FEATURES)


    # Create a new test for predicted data in Model 3a
    N_model3a = sequence_length
    last_N_days_model3a = new_df_model3a[-sequence_length:].values
    last_N_days_scaled_model3a = scaler.transform(last_N_days_model3a)

    test_new_model3a = []
    test_new_model3a.append(last_N_days_scaled_model3a)

    pred_price_scaled_model3a = model.predict(np.array(test_new_model3a))
    pred_price_unscaled_model3a = scaler_pred.inverse_transform(pred_price_scaled_model3a.reshape(-1, 1))

    print(pred_price_unscaled_model3a)

    price_today_model3a = np.round(new_df_model3a['Close'][-1], 2)
    predicted_price_model3a = np.round(pred_price_unscaled_model3a.ravel()[0], 2)
    change_percent_model3a = np.round(100 - (price_today_model3a * 100) / predicted_price_model3a, 2)
    print(price_today_model3a)

    plus = '+'; minus = ''
    print(f'The close price for {stockname} at {end_date} was {price_today_model3a}')
    print(f'The next day predicted close price is {predicted_price_model3a} ({plus if change_percent_model3a > 0 else minus}{change_percent_model3a}%)')

    # Create the line plot graph for Model 3a
    test_df_model3a = pd.DataFrame({'y_test': y_test_unscaled3a.flatten(), 'y_pred': y_pred_model3a.flatten()})
    fig_model3a, ax1_model3a = plt.subplots(figsize=(16, 8), sharex=True)
    sns.lineplot(data=test_df_model3a)
    ax1_model3a.tick_params(axis="x", rotation=0, labelsize=10, length=0)
    plt.title("y_pred vs y_test Truth - Model 3a")
    plt.legend(["y_pred", "y_test"], loc="upper left")
    mpl.rc('hatch', color='k', linewidth=2)
    ax1_model3a.fill_between(test_df_model3a.index, test_df_model3a["y_test"], test_df_model3a["y_pred"], facecolor='white', alpha=.9)
    plt.show()

    # Create the line plot for the residuals in Model 3a
    train_model3a = pd.DataFrame(data_filtered_ext['Close'][:train_data_len + 1]).rename(columns={'Close': 'y_train'})
    valid_model3a = pd.DataFrame(data_filtered_ext['Close'][train_data_len:]).rename(columns={'Close': 'y_test'})
    valid_model3a.insert(1, "y_pred", y_pred_model3a, True)
    valid_model3a.insert(1, "residuals", valid_model3a["y_pred"] - valid_model3a["y_test"], True)
    df_union_model3a = pd.concat([train_model3a, valid_model3a], sort=True)

    df_union_zoom_model3a = df_union_model3a[df_union_model3a.index > display_start_date]

    fig_model3a, ax1_model3a = plt.subplots(figsize=(16, 8))
    plt.title("Original Price vs Predicted Price - Model 3a")
    plt.ylabel(stockname, fontsize=18)
    sns.set_style('white', {'axes.spines.right': False, 'axes.spines.top': False})
    sns.set_palette(["#090364", "#1960EF", "#EF5919"])
    sns.lineplot(data=df_union_zoom_model3a[['y_pred', 'y_train', 'y_test']], linewidth=1.0, dashes=False, ax=ax1_model3a)
    df_sub_model3a = ["#2BC97A" if x > 0 else "#C92B2B" for x in df_union_zoom_model3a["residuals"].dropna()]
    ax1_model3a.bar(height=df_union_zoom_model3a['residuals'].dropna(), x=df_union_zoom_model3a['residuals'].dropna().index, width=3, label='residuals', color=df_sub_model3a)
    plt.legend()
    plt.show()

"""# Model - 4"""

# Round predicted data for Model 4
    y_pred_model4 = np.round(y_pred4)

    df_temp_model4 = df[-sequence_length:]
    new_df_model4 = df_temp_model4.filter(FEATURES)


    # Create a new test for predicted data in Model 4
    N_model4 = sequence_length
    last_N_days_model4 = new_df_model4[-sequence_length:].values
    last_N_days_scaled_model4 = scaler.transform(last_N_days_model4)

    test_new_model4 = []
    test_new_model4.append(last_N_days_scaled_model4)

    pred_price_scaled_model4 = model.predict(np.array(test_new_model4))
    pred_price_unscaled_model4 = scaler_pred.inverse_transform(pred_price_scaled_model4.reshape(-1, 1))

    print(pred_price_unscaled_model4)

    price_today_model4 = np.round(new_df_model4['Close'][-1], 2)
    predicted_price_model4 = np.round(pred_price_unscaled_model4.ravel()[0], 2)
    change_percent_model4 = np.round(100 - (price_today_model4 * 100) / predicted_price_model4, 2)
    print(price_today_model4)

    plus = '+'; minus = ''
    print(f'The close price for {stockname} at {end_date} was {price_today_model4}')
    print(f'The next day predicted close price is {predicted_price_model4} ({plus if change_percent_model4 > 0 else minus}{change_percent_model4}%)')

    # Create the line plot graph for Model 4
    test_df_model4 = pd.DataFrame({'y_test': y_test_unscaled4.flatten(), 'y_pred': y_pred_model4.flatten()})
    fig_model4, ax1_model4 = plt.subplots(figsize=(16, 8), sharex=True)
    sns.lineplot(data=test_df_model4)
    ax1_model4.tick_params(axis="x", rotation=0, labelsize=10, length=0)
    plt.title("y_pred vs y_test Truth - Model 4")
    plt.legend(["y_pred", "y_test"], loc="upper left")
    mpl.rc('hatch', color='k', linewidth=2)
    ax1_model4.fill_between(test_df_model4.index, test_df_model4["y_test"], test_df_model4["y_pred"], facecolor='white', alpha=.9)
    plt.show()

    # Create the line plot for the residuals in Model 4
    train_model4 = pd.DataFrame(data_filtered_ext['Close'][:train_data_len + 1]).rename(columns={'Close': 'y_train'})
    valid_model4 = pd.DataFrame(data_filtered_ext['Close'][train_data_len:]).rename(columns={'Close': 'y_test'})
    valid_model4.insert(1, "y_pred", y_pred_model4, True)
    valid_model4.insert(1, "residuals", valid_model4["y_pred"] - valid_model4["y_test"], True)
    df_union_model4 = pd.concat([train_model4, valid_model4], sort=True)

    df_union_zoom_model4 = df_union_model4[df_union_model4.index > display_start_date]

    fig_model4, ax1_model4 = plt.subplots(figsize=(16, 8))
    plt.title("Original Price vs Predicted Price - Model 4")
    plt.ylabel(stockname, fontsize=18)
    sns.set_style('white', {'axes.spines.right': False, 'axes.spines.top': False})
    sns.set_palette(["#090364", "#1960EF", "#EF5919"])
    sns.lineplot(data=df_union_zoom_model4[['y_pred', 'y_train', 'y_test']], linewidth=1.0, dashes=False, ax=ax1_model4)
    df_sub_model4 = ["#2BC97A" if x > 0 else "#C92B2B" for x in df_union_zoom_model4["residuals"].dropna()]
    ax1_model4.bar(height=df_union_zoom_model4['residuals'].dropna(), x=df_union_zoom_model4['residuals'].dropna().index, width=3, label='residuals', color=df_sub_model4)
    plt.legend()
    plt.show()

"""# Create"""

# CREATE A NEW COLUMN FOR THE ACTUAL UP AND DOWN MOVEMENTS
    y_test_unscaled_df = pd.DataFrame(y_test_unscaled, columns=['Close'])
    y_test_unscaled_df['Actual'] = np.where(y_test_unscaled_df['Close'].diff() > 0, 'Up', 'Down')

    # CREATE A NEW COLUMN FOR THE PREDICTED UP AND DOWN MOVEMENTS
    y_pred_df = pd.DataFrame(y_pred, columns=['Close'])
    y_pred_df['Predicted'] = np.where(y_pred_df['Close'].diff() > 0, 'Up', 'Down')

    # COMPUTE THE CONFUSION MATRIX
    cm = confusion_matrix(y_test_unscaled_df['Actual'], y_pred_df['Predicted'], labels=['Up', 'Down'])

    xticklabels=['Up', 'Down']
    yticklabels=['Up', 'Down']
    # PLOT THE CONFUSION MATRIX AS A HEAT MAP
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=xticklabels, yticklabels=yticklabels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix for Up and Down Movements')

    plt.show()

# COMPUTE THE ACCURACY SCORE
    accuracy = accuracy_score(y_test_unscaled_df['Actual'], y_pred_df['Predicted'])
    print(f'Accuracy: {accuracy:.2f}')

    # COMPUTE THE PRECISION SCORE
    precision = precision_score(y_test_unscaled_df['Actual'], y_pred_df['Predicted'], pos_label='Down')
    print(f'Precision: {precision:.2f}')

    # COMPUTE THE RECALL SCORE
    recall = recall_score(y_test_unscaled_df['Actual'], y_pred_df['Predicted'], pos_label='Down')
    print(f'Recall: {recall:.2f}')

    # COMPUTE THE F1 SCORE
    f1 = f1_score(y_test_unscaled_df['Actual'], y_pred_df['Predicted'], pos_label='Down')
    print(f'F1 score: {f1:.2f}')

    # COMPUTE THE F2 SCORE
    f2 = fbeta_score(y_test_unscaled_df['Actual'], y_pred_df['Predicted'], beta=2, pos_label='Down')
    print(f'F2 score: {f2:.2f}')

# ENCODE THE ACTUAL AND PREDICTED LABELS AS NUMBERS
    # ASSUME 'UP' IS THE POSITIVE CLASS AND 'DOWN' IS THE NEGATIVE CLASS
    y_test_unscaled_num = np.where(y_test_unscaled_df['Actual'] == 'Down', 1, 0)
    y_pred_num = np.where(y_pred_df['Predicted'] == 'Down', 1, 0)

    # COMPUTE THE TPR AND FPR FOR VARIOUS THRESHOLDS
    fpr, tpr, thresholds = roc_curve(y_test_unscaled_num, y_pred_num)

    # COMPUTE THE AREA UNDER THE CURVE (AUC)
    auc = roc_auc_score(y_test_unscaled_num, y_pred_num)

    # PLOT THE ROC CURVE
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Up and Down Movements')

    plt.show()

# COMPUTE THE AREA UNDER THE CURVE (AUC)
    auc = roc_auc_score(y_test_unscaled_num, y_pred_num)
    # PLOT THE ACTUAL AND PREDICTED VALUES AS A SCATTER PLOT
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test_unscaled_df['Close'], y=y_pred_df['Close'], hue=y_test_unscaled_df['Actual'])
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Scatter Plot for Actual and Predicted Values')

    plt.legend()
    plt.show()

"""# Create a table for both actual and predicted values"""

def createTable(original, predict):
        output = '<table border=1 align=left>'
        output += '<tr><th>Original Price</th><th>Predicted Price</th></tr>'
        for i in range(len(original)):
            output += '<tr><td>'+str(original[i])+'</td><td>'+str(predict[i])+'</td></tr>'
        output += '</table>'
        display(HTML(output))
    y_test1 = np.round(y_test_unscaled)

    createTable(y_test1, y_pred,)

import pandas as pd
    from tabulate import tabulate

    # Create a list to store the results for each model
    results = []

    # Model 1

    # Calculate the difference between actual and predicted prices for Model 1
    price_diff_model1 = predicted_price_model - price_today_model

    # Append the results for Model 1 to the list
    results.append([stockname, MSE_model, MAE_model, MEDAE_model, RMSE_model, MAPE_model, MDAPE_model, Acc])

    # Create a DataFrame from the results list
    columns = ['Stock Name', 'MSE', 'MAE', 'MEDAE', 'RMSE', 'MAPE', 'MDAPE', 'R2 Score']
    df_results = pd.DataFrame(results, columns=columns)

    # Model 2

    # Calculate the difference between actual and predicted prices for Model 2
    price_diff_model2 = predicted_price_model2 - price_today_model2

    # Append the results for Model 2 to the list
    results.append([stockname, MSE2, MAE2, MEDAE2, RMSE2, MAPE2, MDAPE2, Acc_model2])

    # Create a DataFrame from the results list
    columns = ['Stock Name', 'MSE', 'MAE', 'MEDAE', 'RMSE', 'MAPE', 'MDAPE', 'R2 Score']
    df_results = pd.DataFrame(results, columns=columns)

    # Model 3

    # Calculate the difference between actual and predicted prices for Model 3
    price_diff_model3 = predicted_price_model3 - price_today_model3

    # Append the results for Model 3 to the list
    results.append([stockname, MSE3, MAE3, MEDAE3, RMSE3, MAPE3, MDAPE3, Acc_model3])

    # Create a DataFrame from the results list
    columns = ['Stock Name', 'MSE', 'MAE', 'MEDAE', 'RMSE', 'MAPE', 'MDAPE', 'R2 Score']
    df_results = pd.DataFrame(results, columns=columns)

    # Model 3a

    # Calculate the difference between actual and predicted prices for Model 3a
    price_diff_model3a = predicted_price_model - price_today_model

    # Append the results for Model 3a to the list
    results.append([stockname, MSE3a, MAE3a, MEDAE3a, RMSE3a, MAPE3a, MDAPE3a, Acc_model3a])

    # Create a DataFrame from the results list
    columns = ['Stock Name', 'MSE', 'MAE', 'MEDAE', 'RMSE', 'MAPE', 'MDAPE', 'R2 Score']
    df_results = pd.DataFrame(results, columns=columns)

    # Model 4

    # Calculate the difference between actual and predicted prices for Model 4
    price_diff_model4 = predicted_price_model - price_today_model

    # Append the results for Model 4 to the list
    results.append([stockname, MSE4, MAE4, MEDAE4, RMSE4, MAPE4, MDAPE4, Acc_model4])

    # Create a DataFrame from the results list
    columns = ['Stock Name', 'MSE', 'MAE', 'MEDAE', 'RMSE', 'MAPE', 'MDAPE', 'R2 Score']
    df_results = pd.DataFrame(results, columns=columns)

    # Convert the DataFrame to a table using tabulate
    table = tabulate(df_results, headers='keys', tablefmt='psql')

    # Display the table
    print(table)

"""# Find how many ups and downs in the timeseries data"""

def count_ups_and_downs(data):
        ups = 0
        downs = 0

        for i in range(1, len(data)):
            if data[i] > data[i - 1]:
                ups += 1
            elif data[i] < data[i - 1]:
                downs += 1

        return ups, downs

dataset = data_filtered_ext['Close']
    ups, downs = count_ups_and_downs(dataset)
    print("Ups:", ups)
    print("Downs:", downs)

def predict_next_point(data):
        ups, downs = count_ups_and_downs(data)
        total = ups + downs
        up_percentage = ups / total
        down_percentage = downs / total

        if up_percentage > down_percentage:
            trend = "up"
        elif up_percentage < down_percentage:
            trend = "down"
        else:
            trend = "unknown"

        if trend == "up":
            prediction = data[-1] + (data[-1] - data[-2])
        elif trend == "down":
            prediction = data[-1] - (data[-2] - data[-1])
        else:
            prediction = data[-1]  # No clear trend, use the last value as the prediction

        return prediction

dataset = data_filtered_ext['Close']
    prediction = predict_next_point(dataset)
    print("Prediction:", prediction)