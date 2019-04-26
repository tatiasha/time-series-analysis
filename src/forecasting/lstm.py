from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

num_classes = 1
batch_size = 32


def lstm(data, train_val):
    # data = pd.read_csv("..//..//tmp_data//sensor_data.csv", header=None)[1].values
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data = scaler.fit_transform(data.reshape(-1, 1))
    data = [i[0] for i in data]
    X_SIZE = 50
    TRAIN_SIZE = int(batch_size*int(int(train_val*len(data))/batch_size))+X_SIZE
    TEST_SIZE = int((len(data) - TRAIN_SIZE)/batch_size)*batch_size

    train_input = np.array(data[:TRAIN_SIZE+1])
    test_input = np.array(data[TRAIN_SIZE-X_SIZE-1:])


    x_train = np.array([np.array([train_input[i:i+X_SIZE]]) for i in range(TRAIN_SIZE-X_SIZE)])
    y_train = np.array([np.array([train_input[i+X_SIZE+1]]) for i in range(TRAIN_SIZE-X_SIZE)])

    x_test = np.array([np.array([test_input[i:i+X_SIZE]]) for i in range(TEST_SIZE)])
    y_test = np.array([np.array([test_input[i+X_SIZE+1]]) for i in range(TEST_SIZE)])



    model = Sequential()
    model.add(LSTM(batch_size, return_sequences=True, stateful=True,
                   batch_input_shape=(batch_size, 1, X_SIZE)))
    model.add(LSTM(batch_size, return_sequences=False, stateful=True))
    # model.add(LSTM(batch_size, stateful=True))
    model.add(Dense(num_classes, activation='relu'))

    model.compile(loss='mae',
                  optimizer='RMSprop',
                  metrics=['mse'])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=30, shuffle=False)
    #
    y_pred = model.predict(x_test, batch_size=batch_size)
    y_pred = scaler.inverse_transform(y_pred)
    y_pred = [i[0] for i in y_pred]

    y_test = scaler.inverse_transform(y_test)
    y_test = [i[0] for i in y_test]

    train_input = [[i] for i in train_input]
    train_input = scaler.inverse_transform(train_input)
    train_input = [i[0] for i in train_input]


    print("MSE: {}".format(np.sqrt(mean_squared_error(y_test, y_pred))))

    plt.plot(range(0, TRAIN_SIZE+1), train_input)
    plt.plot(range(TRAIN_SIZE, TRAIN_SIZE+len(y_pred)), y_test)
    plt.plot(range(TRAIN_SIZE, TRAIN_SIZE+len(y_pred)), y_pred, linestyle='--')
    plt.show()
