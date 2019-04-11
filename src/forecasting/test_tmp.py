from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd
data_dim = 20
timesteps = 1
num_classes = 1
batch_size = 780

# Expected input batch shape: (batch_size, timesteps, data_dim)
# Note that we have to provide the full batch_input_shape since the network is stateful.
# the sample of index i in batch k is the follow-up for the sample i in batch k-1.
model = Sequential()
model.add(LSTM(32, return_sequences=True, stateful=True,
               batch_input_shape=(batch_size, timesteps, data_dim)))
model.add(LSTM(32, return_sequences=True, stateful=True))
model.add(LSTM(32, stateful=True))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Generate dummy training data
# x_train = np.random.random((batch_size * 10, timesteps, data_dim))
# y_train = np.random.random((batch_size * 10, num_classes))
#
# x_test = np.random.random((batch_size * 10, timesteps, data_dim))
# y_test = np.random.random((batch_size * 10, num_classes))
data = pd.read_csv("..//..//tmp_data//sensor_data.csv", header=None)[1].values

TRAIN_SIZE = 800
TEST_SIZE = len(data) - TRAIN_SIZE
X_SIZE = 20

Y_SIZE = 1
train_input = data[:TRAIN_SIZE+1]
test_input = data[TRAIN_SIZE-X_SIZE-1:]

x_train = np.array([np.array([train_input[i:i+X_SIZE]]) for i in range(TRAIN_SIZE-X_SIZE)])
y_train = np.array([np.array([train_input[i+X_SIZE+1]]) for i in range(TRAIN_SIZE-X_SIZE)])

# x_test = np.array([test_input[i:i+X_SIZE] for i in range(TEST_SIZE)])
# y_test = np.array([test_input[i+X_SIZE+1] for i in range(TEST_SIZE)])

# # Generate dummy validation data
# x_val = np.random.random((batch_size * 3, timesteps, data_dim))
# y_val = np.random.random((batch_size * 3, num_classes))

model.fit(x_train, y_train, epochs=5, shuffle=False)

# pred = model.predict(x_test)
# print(np.sqrt(mean_squared_error(y_test, pred)))

