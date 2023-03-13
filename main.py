import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.rcParams["axes.grid"] = True

from lib.utils.perfmets import MSE
from lib.Conventional_NN import network

# CLASSIC ML data set (classification)
df = pd.read_csv("https://raw.githubusercontent.com/practiceprobs/datasets/main/iris/iris.csv")
print(df.head())

data = df.to_numpy()
x = data[:,0:-1].astype(np.float64)
y_class = data[:,-1]
y_1hot = np.zeros((y_class.shape[0],3))
for i in range(y_class.size):
    if y_class[i]=="Iris-setosa":
        y_1hot[i,0] = 1
    elif y_class[i]=="Iris-versicolor":
        y_1hot[i,1] = 1
    elif y_class[i]=="Iris-virginica":
        y_1hot[i,2] = 1

n = network(4,4,3)
n.init_random_weights()
err = n.train(x, y_1hot, 0.01, 10000, batch_size=50, v=True, print_freq=98)
MSE_000 = err[-1]
print("Training MSE = %f" % MSE_000)

epochs = np.arange(err.size)