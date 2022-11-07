# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 16:19:52 2022

@author: Administrator
"""

import pandas as pd

import numpy as np

from keras.models import Sequential

from keras.layers import Dense

from tensorflow.keras.layers import LSTM, Dropout, Dense, Activation

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


df = pd.read_excel("회사 별 검색량, 주가.xlsx", header = 2, sheet_name = '모음', names = ['날짜', '검색총량', '검색량증감률', '주가증감률'], usecols = 'A,B,C,D')
df['날짜'] = pd.to_datetime(df["날짜"])
print(df)



x_train = df['검색총량']
plt.plot(x_train)


y_train = df['주가증감률']
plt.plot(y_train)



x_train, x_test, y_train, y_test = train_test_split\
    # (x_train, y_train, test_size = 0.1)




model = Sequential()
model.add(Dense(units = 64, input_dim = 28*28, activation = 'relu'))
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs = 5000, batch_size = 16)




x_pred = model.predict(x_train)
y_pred = model.predict(y_train)


plt.figure(figsize=(16, 10))
plt.plot(x_train, label = 'search_count')
plt.plot(x_pred, label = 'search_prediction')
plt.legend()
plt.show()



plt.figure(figsize=(16, 10))
plt.plot(y_train, label = 'price_percent')
plt.plot(y_pred, label = 'price_prediction')
plt.legend()
plt.show()



plt.plot(x_train, x_pred)




