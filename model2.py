# saving models as pickle dump and executing models from there


# import dependencies
import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

data = pd.read_csv("student-mat.csv", sep =";")
#print(data.head())

data = data[["G1", "G2", "G3", "studytime", "failures", "absences", "health"]]
#print(data.head())

predict = "G3"
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size= 0.1)
"""
linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print(acc)


with open("studentgrade.pickle", "wb") as f:
    pickle.dump(linear, f)"""

pickle_in = open("studentgrade.pickle", "rb")
linear = pickle.load(pickle_in)
print("Coeff: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)
