from keras.layers.core import Dense
from keras.layers import LSTM
from keras.callbacks import History
import matplotlib.pyplot as plt
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math
import os

#загрузка данных потребителя в большой массив из сохраненного двоичного файда
#также приведем его к диапозону (0,1), изменим форму входных данных на трехмерную
#как того требует наша сеть

data = np.load("LD.npy")
data = data.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1), copy = False)
data = scaler.fit_transform(data)

num_time = 20

x = np.zeros((data.shape[0], num_time))
y = np.zeros((data.shape[0], 1))


for i in range(len(data) - num_time -1):
    x[i] = data[i:i + num_time].T
    y[i] = data[i + num_time +1]

x = np.expand_dims(x, axis=2)

sp = int(0.7 * len(data))

Xtrain, Xtest, Ytrain, Ytest = x[0:sp], x[sp:],y[0:sp],y[sp:]

print(Xtrain.shape, Xtest.shape, Ytrain.shape, Ytest.shape)

hidden = 10
batch = 168
num_epoch = 5
model = Sequential()
#model.add(LSTM(hidden, stateful=True, batch_input_shape=(batch, num_time, 1), return_sequences=False))
model.add(LSTM(168, input_shape=(20, 1), return_sequences=False))
model.add(Dense(1))

model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"])

history = model.fit(Xtrain, Ytrain, batch_size=batch, epochs=num_epoch, validation_data=(Xtest, Ytest), shuffle=False)

# train_size = (Xtrain.shape[0] // batch)* batch
# test_size = (Xtest.shape[0] // batch)* batch
#
# Xtrain, Ytrain = Xtrain[0:train_size], Ytrain[0:train_size]
# Xtest, Ytest = Xtest[0:test_size], Ytest[0:train_size]
#
# print(Xtrain.shape, Xtest.shape, Ytrain.shape, Ytest.shape)
#
# num_epoch = 1
#
# for i in range(num_epoch):
#     print("Epoch {:d}/{:d}".format(i+1, num_epoch))
#     model.fit(Xtrain, Ytrain, batch_size=batch, epochs=1, validation_data=(Xtest, Ytest), shuffle=False)
#     model.reset_states()

scope, _ = model.evaluate(Xtest, Ytest, batch_size=batch)
rmse = math.sqrt(scope)
print("MSE: {:.3f}, RMSE: {:.3f}".format(scope, rmse))



loss_values = history.history['loss']
val_loss_values = history.history['val_loss']

#print[loss_values]

epo = range(1, 6)

plt.plot(epo, loss_values)
plt.plot(epo, val_loss_values)

plt.show()

