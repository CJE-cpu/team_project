# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 19:27:38 2022

@author: Administrator
"""


import pandas as pd # csv 파일 로드
import numpy as np # 행렬 연산
import matplotlib.pyplot as plt # 데이터 시각화
from keras.models import Sequential # deep learning model
from keras.layers import LSTM, Dropout, Dense, Activation
import datetime



data = pd.read_excel("회사 별 검색량, 주가.xlsx", sheet_name = '엔씨소프트', names = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'], usecols = 'AE,AF,AG,AH,AI,AJ,AK')
pd.to_datetime(data['Date'])
data.dropna(inplace=True)


high_prices = data['High'].values
low_prices = data['Low'].values
mid_prices = (high_prices + low_prices)/2


# 최근 50일 간의 데이터 확인
seq_len = 50 # window size
sequence_length = seq_len + 1 

result = []
for index in range(len(mid_prices) - sequence_length):
    result.append(mid_prices[index: index + sequence_length])
    
    
    
normalized_data = []
for window in result:
    normalized_window = [((float(p) / float(window[0]))-1) for p in window]
    normalized_data.append(normalized_window)

result = np.array(normalized_data)

# split train and test data
row = int(round(result.shape[0]*0.9)) 
train = result[:row, :]
np.random.shuffle(train) # training set shuffle

x_train = train[:, :-1] 
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
y_train = train[:, -1]

x_test = result[row:, :-1] # 앞에 50개
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
y_test = result[row:, -1] # 뒤에 1개

x_train.shape, x_test.shape




model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(50,1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(1, activation='linear')) # 다음날 하루의 데이터를 예측함
model.compile(loss='mse', optimizer='rmsprop') # Loss function
model.summary()




# Functional API
from tensorflow.keras.layers import LSTM, Dropout, Input, Dense, Activation
from tensorflow.keras.models import Model # Functional API 를 쓰기위한 Model class
input_tensor = Input(shape=(50,1))
# FunctionalAPI(레이어를 만드는 주요 파라미터)(입력 인자)
x = LSTM(50, return_sequences=True)(input_tensor)
x = LSTM(64, return_sequences=False)(x)
output = Dense(1, activation='linear')(x)
model = Model(inputs=input_tensor, outputs=output) # input 인자와 output 인자가 필요함

model.compile(loss='mse', optimizer='Adam')
model.summary()


model.fit(x_train, y_train,
         validation_data=(x_test, y_test),
         batch_size=10, # 10개씩 묶어서 학습시킨다.
         epochs=20) # 20번동안 반복


pred = model.predict(x_test) # 테스트 데이터 예측

fig = plt.figure(facecolor='white')
ax = fig.add_subplot(111)
ax.plot(y_test, label='True')
ax.plot(pred, label='Prediction')
ax.legend()
plt.show()


# ------------------------------------------------------------------------------------------------------------------------






