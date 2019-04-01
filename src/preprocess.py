from os import listdir
from os.path import isfile, join
import pandas as pd
import matplotlib.pyplot as plt
import forecasting.statistical as f_stat
import outliers.statistical as o_stat
import clustering.km as c_stat


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


def main():
    path = "..//sample_data"
    files = get_files_names(path)
    files_names_info = parse_files_names(files)
    print_files_info(files_names_info)
    sensors_ids, periods = choose_file(files_names_info)
    for f in range(len(sensors_ids)):
        start_ = str(periods[f].split("-")[0])
        end_ = str(periods[f].split("-")[1])
        file_info_ = files_names_info.loc[files_names_info['id'] == str(sensors_ids[f])]
        file_info_ = file_info_.loc[file_info_['start_p'] == start_]
        file_info_ = file_info_.loc[file_info_['end_p'] == end_]
        data_ = pd.read_csv("{}/{}".format(path, file_info_['file_name'].values[0]), header=None)
        print(data_.head())

    data_ = data_[:1000]

    f_stat.ARIMA_model(data_, plot=1, start_point=int(0.9*len(data_)), end_point=int(1.1*len(data_)))
    o_stat.outlier_detection(data_)
    c_stat.clustering_kmeans(data_)


if __name__ == "__main__":
    main()
