import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn import tree
from keras import Sequential

data = pd.read_csv("..//..//tmp_data//sensor_data.csv", header=None)[1].values

TRAIN_SIZE = 800
TEST_SIZE = len(data) - TRAIN_SIZE
X_SIZE = 20

Y_SIZE = 1
train_input = data[:TRAIN_SIZE+1]
test_input = data[TRAIN_SIZE-X_SIZE-1:]

X_train = [train_input[i:i+X_SIZE] for i in range(TRAIN_SIZE-X_SIZE)]
Y_train = [train_input[i+X_SIZE+1] for i in range(TRAIN_SIZE-X_SIZE)]

X_test = [test_input[i:i+X_SIZE] for i in range(TEST_SIZE)]
Y_test = [test_input[i+X_SIZE+1] for i in range(TEST_SIZE)]

print(len(Y_train))
print(len(Y_test))
print(len(data))

clf = tree.DecisionTreeRegressor()
clf = clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)

rmse = np.sqrt(mean_squared_error(Y_test, y_pred))
plt.plot(range(0, TRAIN_SIZE+1), train_input)
plt.plot(range(TRAIN_SIZE, TRAIN_SIZE+len(y_pred)), Y_test)
plt.plot(range(TRAIN_SIZE, TRAIN_SIZE+len(y_pred)), y_pred, linestyle='--')
# plt.plot(range(len(data)), data, c='red', linewidth=0.5)
plt.show()
print(rmse)


