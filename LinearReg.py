__author__ = 'tr1c4011'

import pandas as pd
import numpy as np
import operations as op
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

all_data = pd.read_csv('Afeb.csv' )
all_data = all_data.values

test_data = all_data[0:2, :]
training_data = all_data[3:18,:]

training_Xs = training_data[:,0:2]
trainin_poly = training_Xs
training_Xs[:,1] = np.power(training_Xs[:,1], 2)
#training_Xs[:,0] = np.power(training_Xs[:,0], 2)
training_Y = training_data[:,2]
ones = np.ones((15,1))
training_Xs = np.append(ones, training_Xs, 1)

test_Xs = test_data[:,0:2]
test_Ys = test_data[:,2]

param_vector = [15,15,15]
param_vector = np.transpose(param_vector)

loss = op.loss_function(param_vector, training_Xs, training_Y)
print(loss)
teta = op.derivative(training_Xs, training_Y)
print(teta)
loss = op.loss_function(teta, training_Xs, training_Y)
print(loss)


# pass the order of your polynomial here
poly = PolynomialFeatures(3)

# convert to be used further to linear regression
X_transform = poly.fit_transform(trainin_poly)

# create a Linear Regressor
lin_regressor = LinearRegression()
lin_regressor.fit(X_transform, training_Y)
print(lin_regressor.coef_)








