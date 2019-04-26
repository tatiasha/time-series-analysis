from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

num_classes = 1
batch_size = 32


def lstm(data_1, data_2, train_val):
    # data = pd.read_csv("..//..//tmp_data//sensor_data.csv", header=None)[1].values
    x_value = data_1
    y_value = data_2
    values = np.array([*x_value, *y_value])

    scaler = MinMaxScaler(feature_range=(-10, 10))
    data = scaler.fit_transform(values.reshape(-1, 1))
    data = [i[0] for i in data]

    X_SIZE = 50
    TRAIN_SIZE = int(batch_size*int(int(train_val*len(x_value))/batch_size))+X_SIZE
    TEST_SIZE = int((len(x_value) - TRAIN_SIZE)/batch_size)*batch_size

    train_input = np.array(data[:TRAIN_SIZE])
    train_output = np.array(data[len(x_value):len(x_value)+TRAIN_SIZE])

    test_input = np.array(data[TRAIN_SIZE:len(x_value)])
    test_output = np.array(data[TRAIN_SIZE+len(x_value):])

    x_train = np.array([[np.array(train_input[i:i+X_SIZE])] for i in range(TRAIN_SIZE-X_SIZE)])
    y_train = np.array([np.array([train_output[i]]) for i in range(TRAIN_SIZE-X_SIZE)])

    x_test = np.array([[np.array(test_input[i:i+X_SIZE])] for i in range(TEST_SIZE-X_SIZE)])
    x_test = x_test[:batch_size*int(len(x_test)/batch_size)]
    print(x_test.shape)

    y_test = np.array([np.array([test_output[i]]) for i in range(TEST_SIZE-X_SIZE)])
    y_test = y_test[:batch_size*int(len(x_test)/batch_size)]
    print(y_test.shape)

    model = Sequential()
    model.add(LSTM(batch_size, return_sequences=True, stateful=True,
                   batch_input_shape=(batch_size, 1, X_SIZE)))
    model.add(LSTM(batch_size, return_sequences=True, stateful=True))
    model.add(LSTM(batch_size, stateful=True))
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

    input = scaler.inverse_transform([data[:len(x_value)]])
    input = input[0]

    output = scaler.inverse_transform([data[len(x_value):]])
    output = output[0]

    print("MSE: {}".format(np.sqrt(mean_squared_error(y_test, y_pred))))

    plt.plot(range(len(input)), input, label="input")
    plt.plot(range(len(output)), output, label="output")

    # plt.plot(range(TRAIN_SIZE, TRAIN_SIZE+len(test_input)), test_input, label="test_input")
    # plt.plot(range(TRAIN_SIZE, TRAIN_SIZE+len(test_output)), test_output, label="test_output")


    # plt.plot(range(TRAIN_SIZE, TRAIN_SIZE+len(y_test)), y_test, label='y_test')
    plt.plot(range(TRAIN_SIZE, TRAIN_SIZE+len(y_pred)), y_pred, linestyle='--', label='y_pred')
    plt.legend()
    plt.show()
