from os import listdir
from os.path import isfile, join
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import forecasting.statistical as f_stat
from forecasting import lstm, lstm_2signal
import outliers.statistical as o_stat
import clustering.km as c_stat
from scipy.stats.stats import pearsonr
import statsmodels.api as sm
import random

def get_files_names(path):
    return [i for i in listdir(path) if isfile(join(path, i))]


def parse_files_names(files):
    data = pd.DataFrame(columns=["file_name", "type", "id", "start_p", "end_p"])
    for i in files:
        info = i[:-4].split("-")
        data = data.append({"file_name": i, "type": info[0], "id": info[1], "start_p": info[2], "end_p": info[3]},
                           ignore_index=True)
    return data


def print_files_info(data):
    print("=======")
    periods = [x+"-"+y for x, y in zip(data['start_p'].values, data['end_p'].values)]
    print("{} unique sensors".format(len(data["id"].unique())))
    print("{} unique periods".format(len(set(periods))))
    for i in data['id'].unique():
        tmp_sen_data = data.loc[data['id'] == i]
        periods = [x + "-" + y for x, y in zip(tmp_sen_data['start_p'].values, tmp_sen_data['end_p'].values)]
        print("Sensor {} - Periods {}".format(i, len(periods)))
    print("=======")


def choose_file(data,  default_flag=1, n_sensors=1, tmp_sen=1, tmp_per_id=0):
    id_sen = data['id'].unique()
    sensors = []
    sensors_per = []

    if default_flag != 1:
        n_sensors = input("Set number of sensors up to {} \n".format(len(id_sen)))
        for i in range(int(n_sensors)):
            tmp_sen = input("Choose id of {} sensor {} \n".format(i, id_sen))

            if tmp_sen not in id_sen:
                print("Incorrect id")
                break
            else:
                sensors.append(tmp_sen)

            tmp_sen_data = data.loc[data['id'] == tmp_sen]
            periods = [x+"-"+y for x, y in zip(tmp_sen_data['start_p'].values, tmp_sen_data['end_p'].values)]
            tmp_per_id = int(input("Choose period of {} sensor {} \n".format(i, periods)))
            sensors_per.append(periods[tmp_per_id])
    else:
        print("Number of sensors {}".format(n_sensors))
        print("id of sensor {}".format(tmp_sen))
        sensors.append(tmp_sen)
        tmp_sen_data = data.loc[data['id'] == str(tmp_sen)]
        periods = [x + "-" + y for x, y in zip(tmp_sen_data['start_p'].values, tmp_sen_data['end_p'].values)]
        sensors_per.append(periods[tmp_per_id])
        print("Period of sensor {}".format(periods[tmp_per_id]))

    return sensors, sensors_per


def correlation(signal_1, signal_2):
    return pearsonr(signal_1, signal_2)


def main():
    # path = "..//sample_data"
    # files = get_files_names(path)
    # files_names_info = parse_files_names(files)
    # print_files_info(files_names_info)
    # sensors_ids, periods = choose_file(files_names_info)
    # for f in range(len(sensors_ids)):
    #     start_ = str(periods[f].split("-")[0])
    #     end_ = str(periods[f].split("-")[1])
    #     file_info_ = files_names_info.loc[files_names_info['id'] == str(sensors_ids[f])]
    #     file_info_ = file_info_.loc[file_info_['start_p'] == start_]
    #     file_info_ = file_info_.loc[file_info_['end_p'] == end_]
    #     data_ = pd.read_csv("{}/{}".format(path, file_info_['file_name'].values[0]), header=None)
    #     print(data_.head())
    #
    # data_ = data_[:10000]

    M = 100000
    data_0 = pd.read_csv("..//data//sens0-0.csv", header=None, delimiter=';')[1].values[:M]
    data_1 = pd.read_csv("..//data//sens0-1.csv", header=None, delimiter=';')[1].values[:M]
    data_2 = pd.read_csv("..//data//sens0-2.csv", header=None, delimiter=';')[1].values[:M]
    data_3 = pd.read_csv("..//data//msen0-0.csv", header=None, delimiter=';')[1].values[:M]
    data_4 = pd.read_csv("..//data//msen1-0.csv", header=None, delimiter=';')[1].values[:M]
    # f_stat.ARIMA_model(data_, plot=1, start_point=int(0.85*len(data_)), end_point=int(len(data_)))
    # o_stat.outlier_detection(data_, .005)
    # c_stat.clustering_kmeans(data_)
    lstm_2signal.lstm(data_0, data_3, 0.85)
    # lstm.lstm(data_1[1].values, 0.85)


def generate_signal():
    N = 5000
    A_1 = [i % 150 for i in range(N)]
    noise = [random.uniform(-10, 10) for _ in range(N)]
    signal_0 = [A_1[i]*np.sin(50*i)+noise[i] for i in range(N)]

    signal_1 = [np.tan(i) + noise[i] for i in range(N)]
    signal_2 = [A_1[i]*np.cos(50*i) + noise[i] for i in range(N)]
    signal_3 = [A_1[i]/3*np.sin(5*i) + random.uniform(-5, 5) for i in range(N)]

    data = pd.DataFrame(data=signal_0)
    data.to_csv("..//tmp_data//sensor_0.csv", header=None)

    data = pd.DataFrame(data=signal_1)
    data.to_csv("..//tmp_data//sensor_1.csv", header=None)

    data = pd.DataFrame(data=signal_2)
    data.to_csv("..//tmp_data//sensor_2.csv", header=None)

    data = pd.DataFrame(data=signal_3)
    data.to_csv("..//tmp_data//sensor_3.csv", header=None)

    plt.subplot(221)
    plt.title("signal 0")
    plt.plot(signal_0)

    plt.subplot(222)
    plt.title("signal 1")
    plt.plot(signal_1)

    plt.subplot(223)
    plt.title("signal 2")
    plt.plot(signal_2)

    plt.subplot(224)
    plt.title("signal 3")
    plt.plot(signal_3)

    plt.show()


if __name__ == "__main__":
    # main()
    M = 100000
    data_0 = pd.read_csv("..//data//sens0-0.csv", header=None, delimiter=';')[1].values[:M]
    data_1 = pd.read_csv("..//data//sens0-1.csv", header=None, delimiter=';')[1].values[:M]
    data_2 = pd.read_csv("..//data//sens0-2.csv", header=None, delimiter=';')[1].values[:M]
    data_3 = pd.read_csv("..//data//msen0-0.csv", header=None, delimiter=';')[1].values[:M]
    data_4 = pd.read_csv("..//data//msen1-0.csv", header=None, delimiter=';')[1].values[:M]
    # #
    # #
    # # # autocorr = np.correlate(data_0, data_3, mode='full')
    # # # autocorr_max = max(autocorr)
    # # # autocorr_norm = [i/autocorr_max for i in autocorr]
    # # # plt.plot(autocorr_norm)
    # # #
    #
    #
    # # autocorr = np.correlate(data_1, data_2, mode='full')
    # # autocorr_max = max(autocorr)
    # # autocorr_norm = [i / autocorr_max for i in autocorr]
    # # plt.plot(autocorr_norm)
    # # plt.show()
    #
    # # fig = sm.graphics.tsa.plot_acf(data_0, ax=ax, lags=10)
    # # plt.savefig("acf.png")
    # #
    #
    # #
    # # #
    #
    print("Correlation of sensor {} and sensor {}: {}".format(0, 1, correlation(data_0, data_1)))
    print("Correlation of sensor {} and sensor {}: {}".format(0, 2, correlation(data_0, data_2)))
    print("Correlation of sensor {} and sensor {}: {}".format(0, 3, correlation(data_0, data_3)))
    print("Correlation of sensor {} and sensor {}: {}".format(0, 4, correlation(data_0, data_4)))

    print("Correlation of sensor {} and sensor {}: {}".format(1, 2, correlation(data_1, data_2)))
    print("Correlation of sensor {} and sensor {}: {}".format(1, 3, correlation(data_1, data_3)))
    print("Correlation of sensor {} and sensor {}: {}".format(1, 4, correlation(data_1, data_4)))

    print("Correlation of sensor {} and sensor {}: {}".format(2, 3, correlation(data_2, data_3)))
    print("Correlation of sensor {} and sensor {}: {}".format(2, 4, correlation(data_2, data_4)))

    print("Correlation of sensor {} and sensor {}: {}".format(3, 4, correlation(data_3, data_4)))
    plt.plot(data_1)
    plt.plot(data_2)
    plt.show()
    # #
    # #
    # #
    # #
    # # # data.to_csv("..//tmp_data//sensor_data.csv", index=None, header=None)
    # # # generate_signal()
    #
