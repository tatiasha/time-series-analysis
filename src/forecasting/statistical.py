from pandas import read_csv
from pandas import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error


def ARIMA_model(data, p=1, d=1, q=0, plot=0, start_point=0, end_point=1):
    Y = data[1].values
    if start_point < len(Y):
        train, test = Y[0:start_point], Y[start_point-1:end_point]
    else:
        train, test = Y, []
    history = [x for x in train]
    predictions = list()
    for t in range(end_point-start_point+1):
        model = ARIMA(history, order=(p, d, q))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()[0]
        predictions.append(output)
        if t < len(test):
            history.append(test[t])
        else:
            history.append(output)

    if start_point < len(Y):
        error = mean_squared_error(test, predictions[:len(test)])
        print('Test MSE: {}'.format(error))
    else:
        print('MSE can not be calculated')

    if plot == 1:
        x_init = [i for i in range(0, len(Y))]
        x_prediction = [i for i in range(start_point-1, end_point)]
        plt.plot(x_init, Y, color='blue', label="Initial values")
        plt.plot(x_prediction, predictions, color='red', label="Predicted values")
        plt.legend()
        plt.show()

