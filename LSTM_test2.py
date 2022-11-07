# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 09:05:35 2022

@author: sang2
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential

raw_df = pd.read_excel('C:/Users/sang2/pythonworkspace/data/하이브 검색량 및 주가_빈공간삭제.xlsx')
raw_df.head()

plt.figure(figsize = (7,4))
plt.title('HYBE STOCK PRICE')
plt.ylabel('price')
plt.xlabel('period')
plt.grid()

plt.plot(raw_df['sum'], label = 'sum', color = 'b')
plt.legend(loc = 'best')

plt.show()

#데이터 전처리

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scale_cols = ['sum', 'bts'] #정규화 대상

scaled_df = scaler.fit_transform(raw_df[scale_cols]) #정규화 수행

print(type(scaled_df), '\n') 

scaled_df = pd.DataFrame(scaled_df, columns = scale_cols) #정규화된 새로운 Dataframe 생성

feature_cols = ['sum', 'bts']
label_cols = ['sum']

label_df = pd.DataFrame(scaled_df, columns = label_cols)
feature_df = pd.DataFrame(scaled_df, columns = feature_cols)

print(feature_df)
print(label_df)

label_np = label_df.to_numpy()
feature_np = feature_df.to_numpy()

window_size = 1

def make_sequene_dataset(feature, label, window_size):
    feature_list = [] #생성될 feature list
    label_list = [] #생성될 label list
    
    for i in range(len(feature)- window_size):
        feature_list.append(feature[i: 1+window_size])
        label_list.append(label[i+window_size])
        
    return np.array(feature_list), np.array(label_list)

X, Y = make_sequene_dataset(feature_np, label_np, window_size)

print(X.shape, Y.shape)

#입력 파라미터 feature, label => numpy type

split = -200

x_train = X[0:split]
y_train = Y[0:split]

x_test = X[split]
y_test = Y[split]

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

model = Sequential()
model.add(LSTM(128, activation = 'tanh', input_shape = x_train[0].shape))
model.add(Dense(1, activation = 'linear'))
model.summary()


model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 100, batch_size = 16)
