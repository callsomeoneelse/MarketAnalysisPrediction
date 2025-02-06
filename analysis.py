import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, SimpleRNN, GRU
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from yahoo_fin import stock_info as si
from collections import deque
from sklearn import preprocessing
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf
from textblob import TextBlob

import os
import numpy as np
import pandas as pd
import random
import time


np.random.seed(200)
tf.random.set_seed(200)
random.seed(200)

N_STEPS = 50
LOOKUP_STEP = 30
SCALE = True
scale_str = f"sc-{int(SCALE)}"
SHUFFLE = True
shuffle_str = f"sh-{int(SHUFFLE)}"
SPLIT_BY_DATE = False
split_by_date_str = f"sbd-{int(SPLIT_BY_DATE)}"
date_now = time.strftime("%Y-%m-%d")
FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]
TEST_SIZE = 0.2

# model parameters
N_LAYERS = 4
CELL = GRU
UNITS = 256
DROPOUT = 0.4
BIDIRECTIONAL = True
LOSS = "mean_squared_error"
OPTIMIZER = "rmsprop"
BATCH_SIZE = 128
EPOCHS = 15

ticker = "TSLA"
ticker_data_filename = os.path.join("data", f"{ticker}_{date_now}.csv")
model_name = f"{date_now}_{ticker}-{shuffle_str}-{scale_str}-{split_by_date_str}-\
{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"
if BIDIRECTIONAL:
    model_name += "-b"

if not os.path.isdir("results"):
    os.mkdir("results")
if not os.path.isdir("logs"):
    os.mkdir("logs")
if not os.path.isdir("data"):
    os.mkdir("data")

##############################################################################################################


def shuffle_in_unison(a, b):
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)

# load_data function from P1


def load_data(ticker, n_steps=50, scale=True, shuffle=True, lookup_step=1, split_by_date=True,
              test_size=0.2, feature_columns=['adjclose', 'volume', 'open', 'high', 'low']):
    if isinstance(ticker, str):
        df = si.get_data(ticker)
    elif isinstance(ticker, pd.DataFrame):
        df = ticker
    else:
        raise TypeError(
            "ticker can be either a str or a `pd.DataFrame` instances")
    result = {}
    result['df'] = df.copy()
    for col in feature_columns:
        assert col in df.columns, f"'{col}' does not exist in the dataframe."
    if "date" not in df.columns:
        df["date"] = df.index
    if scale:
        column_scaler = {}
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(
                np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler
        result["column_scaler"] = column_scaler
    df['future'] = df['adjclose'].shift(-lookup_step)
    last_sequence = np.array(df[feature_columns].tail(lookup_step))
    df.dropna(inplace=True)
    sequence_data = []
    sequences = deque(maxlen=n_steps)
    for entry, target in zip(df[feature_columns + ["date"]].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])
    last_sequence = list([s[:len(feature_columns)]
                         for s in sequences]) + list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)
    result['last_sequence'] = last_sequence
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)
    X = np.array(X)
    y = np.array(y)
    if split_by_date:
        train_samples = int((1 - test_size) * len(X))
        result["X_train"] = X[:train_samples]
        result["y_train"] = y[:train_samples]
        result["X_test"] = X[train_samples:]
        result["y_test"] = y[train_samples:]
        if shuffle:
            shuffle_in_unison(result["X_train"], result["y_train"])
            shuffle_in_unison(result["X_test"], result["y_test"])
    else:
        result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y,
                                                                                                    test_size=test_size, shuffle=shuffle)
    dates = result["X_test"][:, -1, -1]
    result["test_df"] = result["df"].loc[dates]
    result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(
        keep='first')]
    result["X_train"] = result["X_train"][:, :,
                                          :len(feature_columns)].astype(np.float32)
    result["X_test"] = result["X_test"][:, :,
                                        :len(feature_columns)].astype(np.float32)
    return result


def train_LSTM(sequence_length, n_features, units=256, cell=LSTM, n_layers=2, dropout=0.3,
               loss="mean_squared_error", optimizer="rmsprop", bidirectional=False):

    model = Sequential()  # Create instance of keras sequential model

    for i in range(n_layers):  # loop for number of layers defined
        if i == 0:  # first layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True,
                          batch_input_shape=(None, sequence_length, n_features))))
            else:
                model.add(cell(units, return_sequences=True,
                          batch_input_shape=(None, sequence_length, n_features)))
        elif i == n_layers - 1:  # last layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=False)))
            else:
                model.add(cell(units, return_sequences=False))
        else:
            # hidden layers
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))

    model.add(Dropout(dropout))  # add dropout

    model.add(Dense(1, activation="linear"))

    model.compile(loss=loss, metrics=[
                  "mean_squared_error"], optimizer=optimizer)

    return model


def train_SARIMA(data, lookup_step=LOOKUP_STEP):
    df = data["df"].copy()  # create a copy of the pandas dataframe

    # define the series variable as the adjclose feature column
    series = df["adjclose"]

    p, d, q = 3, 2, 5  # define SARIMA hyperparameters for order

    P, D, Q, s = 2, 1, 2, 12  # define SARIMA hyperparameters for seasonal order

    sarima_model = sm.tsa.statespace.SARIMAX(
        series, order=(p, d, q), seasonal_order=(P, D, Q, s))  # train the model

    results = sarima_model.fit(disp=True)  # fit the model to data

    prediction = results.get_forecast(steps=lookup_step)  # get prediction
    predicted_mean = prediction.predicted_mean  # get mean value for predicttion

    return predicted_mean.iloc[-1]


def train_ARIMA(data, lookup_step=LOOKUP_STEP):
    df = data["df"].copy()  # create a copy of the pandas dataframe

    # define the series variable as the adjclose feature column
    series = df["adjclose"]

    p, d, q = 3, 2, 5  # define ARIMA hyperparameters

    model = ARIMA(series, order=(p, d, q))  # train the model
    results = model.fit()  # fit the model to data

    prediction = results.forecast(steps=lookup_step)  # get prediction

    return prediction.iloc[-1]


data = load_data(ticker, N_STEPS, scale=SCALE, split_by_date=SPLIT_BY_DATE,
                 shuffle=SHUFFLE, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE,
                 feature_columns=FEATURE_COLUMNS)

data["df"].to_csv(ticker_data_filename)

model = train_LSTM(N_STEPS, len(FEATURE_COLUMNS), loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                   dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=False)


checkpointer = ModelCheckpoint(os.path.join(
    "results", model_name + ".h5"), save_weights_only=True, save_best_only=True, verbose=1)
tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))

history = model.fit(data["X_train"], data["y_train"],
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(data["X_test"], data["y_test"]),
                    callbacks=[checkpointer, tensorboard],
                    verbose=1)


def final_dataframe(model, data):
    X_test = data["X_test"]  # extract testing dataset
    Y_test = data["y_test"]

    # uses provided model to make predictions on the test dataset
    y_prediction = model.predict(X_test)

    # inversely scale y_test and y_prediction to get true price values
    if SCALE:
        Y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(
            np.expand_dims(Y_test, axis=0)))
        y_prediction = np.squeeze(
            data["column_scaler"]["adjclose"].inverse_transform(y_prediction))

    test_df = data["test_df"]  # retrieve testing dataframe
    # predicted prices added to dataframe
    test_df[f"adjclose_{LOOKUP_STEP}"] = y_prediction
    # true prices added to dataframe
    test_df[f"true_adjclose_{LOOKUP_STEP}"] = Y_test

    test_df.sort_index(inplace=True)  # sort dataframe by index
    final_df = test_df

    return final_df


def predict(model, data):
    # retrieve the last N steps of the sequence
    last_sequence = data["last_sequence"][-N_STEPS:]

    # increase dimensions od the last_sequence array
    last_sequence = np.expand_dims(last_sequence, axis=0)

    # predicts the price based on last sequence
    prediction = model.predict(last_sequence)

    # If the data was previously scaled (indicated by the SCALE variable), the code uses an inverse scaler (specifically for the "adjclose" column)
    # to transform the prediction back to its original scale to get the predicted price in actual value.
    if SCALE:
        predicted_price = data["column_scaler"]["adjclose"].inverse_transform(prediction)[
            0][0]
    else:
        predicted_price = prediction[0][0]

    return predicted_price


def get_sentiment(newstitle):

    analysis = TextBlob(newstitle)

    # get sentiment polarity
    return analysis.sentiment.polarity


def get_news(ticker):

    # get market news data for necessary ticker
    news = yf.Ticker(ticker).news

    # get sentiment polarity for each news 'title'
    sentiments = [get_sentiment(item['title']) for item in news]

    # return average polarity of news surrounding the specified ticker
    return sum(sentiments) / len(sentiments)


def ensemble(model, sarima_data, lstm_data, arima_data, ticker):
    lstm_prediction = predict(model, lstm_data)
    sarima_prediction = train_SARIMA(sarima_data)
    arima_prediction = train_ARIMA(arima_data)

    ensemble_prediction = (
        lstm_prediction + sarima_prediction + arima_prediction) / 3  # average the prediction values

    news_sentiment = get_news(ticker)
    adjustment = 1 + news_sentiment  # account for negative polarity

    final_prediction = ensemble_prediction * adjustment

    return final_prediction


final_df = final_dataframe(model, data)
future_price = ensemble(model, data, data, data, "TSLA")  # use TSLA ticker

print(f"Future price after {LOOKUP_STEP} days is {future_price:.2f}$")
