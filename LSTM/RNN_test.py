# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# 샘플 데이터 만드는 코드
import random
import sys
from tensorflow import keras
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import string
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt

def generate_time_series():
    string_pool = ["A", "B", "C", "D", "E", "F"]
    char_to_index = dict((char, index) for index, char in enumerate(string_pool)) # 글자에 고유한 정수 인덱스 부여
    index_to_char = {}

    for key, value in char_to_index.items():
        index_to_char[value] = key
    
    array = [[0 for col in range(30)] for row in range(2000)]
    for i in range(2000):
        for j in range(30):
            a = random.choice(string_pool)
            num = char_to_index[a]
            array[i][j] = num
        array[i].sort()

    for i in range(1000):
        array[i].reverse()
    arr = np.array(array)
    np.random.shuffle(arr)
    arr = np.array(arr)
    return arr
    
series = generate_time_series()

# 앞의 15개 좌표를 이용해 뒤의 15개 좌표 예측하기 위한 데이터셋
X_train, Y_train = series[:1000, :15], series[:1000, -15:]
X_train = keras.utils.to_categorical(X_train, 6)
Y_train = keras.utils.to_categorical(Y_train, 6)
X_train = np.array(X_train)
X_valid, Y_valid = series[1000:1500, :15], series[1000:1500, -15:]
X_valid = keras.utils.to_categorical(X_valid, 6) # 원 핫 인코딩
Y_valid = keras.utils.to_categorical(Y_valid, 6) # 원 핫 인코딩

# 테스트 데이터셋
test = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5] 
test = np.array(test)
X_test = np.split(test, 2)[0]
Y_test = np.split(test, 2)[1]
X_test = keras.utils.to_categorical(X_test, 6) # 원 핫 인코딩
X_test = np.array(X_test)

# +
# LSTM 이용한 학습 및 예측 확인
import random
import sys
from tensorflow import keras
import numpy as np
import string
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed


# LSTM 모델 생성 및 학습
model = Sequential()
model.add(LSTM(256, input_shape=(15, 6), return_sequences=True))
model.add(LSTM(256, return_sequences=True))
model.add(TimeDistributed(Dense(6, activation='softmax')))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
early_stopping_cb = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
history = model.fit(X_train, Y_train, epochs=100, validation_data=(X_valid, Y_valid), callbacks=[early_stopping_cb])
model.evaluate(X_valid, Y_valid)
model.summary()

# 예측 결과와 실제 값 시각화
X = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
X_test = X_test.reshape(1,15,6)
Y_pred = model.predict(X_test)
Y_pred = np.argmax(Y_pred, axis=2)
print(Y_pred)
plt.scatter(X, Y_pred, c='green')
plt.scatter(X, Y_test, c='red')
plt.title('Test')
plt.ylabel('location')
plt.xlabel('time')
plt.legend(['prediction', 'real'], loc='lower right')
plt.show()

# 정확도 시각화
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Training accuracy', 'Validation accuracy'], loc='lower right')
plt.show()

# loss 시각화
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Training loss', 'Validation loss'], loc='upper right')
plt.show()

# +
# SimpleRNN 이용한 학습 및 예측 확인
import random
import sys
from tensorflow import keras
import numpy as np
import string
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed, SimpleRNN

# SimpleRNN 모델 생성 및 학습
model = Sequential()
model.add(SimpleRNN(256, input_shape=(15, 6), return_sequences=True))
model.add(SimpleRNN(256, return_sequences=True))
model.add(TimeDistributed(Dense(6, activation='softmax')))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
early_stopping_cb = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
history = model.fit(X_train, Y_train, epochs=100, validation_data=(X_valid, Y_valid), callbacks=[early_stopping_cb])
model.evaluate(X_valid, Y_valid)
model.summary()

# 예측 결과와 실제 값 시각화
X = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
X_test = X_test.reshape(1,15,6)
Y_pred = model.predict(X_test)
Y_pred = np.argmax(Y_pred, axis=2)
print(Y_pred)
plt.scatter(X, Y_pred, c='green')
plt.scatter(X, Y_test, c='red')
plt.title('Test')
plt.ylabel('location')
plt.xlabel('time')
plt.legend(['prediction', 'real'], loc='lower right')
plt.show()

# 정확도 시각화
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Training accuracy', 'Validation accuracy'], loc='lower right')
plt.show()

# loss 시각화
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Training loss', 'Validation loss'], loc='upper right')
plt.show()
# -


