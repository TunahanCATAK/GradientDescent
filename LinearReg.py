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
training_Y = training_data[:,2]
ones = np.ones((15,1))
training_Xs = np.append(ones, training_Xs, 1)


test_Xs = test_data[:,0:2]
test_Ys = test_data[:,2]

param_vector = [0.1,0.1,0.1]
param_vector = np.transpose(param_vector)

loss = op.loss_function(param_vector, training_Xs, training_Y)
print(loss)
teta = op.derivative(training_Xs, training_Y)
print(teta)
loss = op.loss_function(teta, training_Xs, training_Y)
print(loss)


# pass the order of your polynomial here
poly = PolynomialFeatures(2)

# convert to be used further to linear regression
X_transform = poly.fit_transform(trainin_poly)

# create a Linear Regressor
lin_regressor = LinearRegression()
lin_regressor.fit(X_transform, training_Y)
print(lin_regressor.coef_)
loss = op.loss_function(lin_regressor.coef_, X_transform, training_Y)
print("loss")
print(loss)

#create a non-linear model
#fi(x) = [1, x1, x2, x1^2, x2^2] --> nando de freitas, lecture4
X1_squares = np.power(training_Xs[:,1], 2)
X2_squares = np.power(training_Xs[:,2], 2)

X1_squares = X1_squares.reshape(15,1)
X2_squares = X2_squares.reshape(15,1)

training_Xs = np.append(training_Xs, X1_squares, 1)
training_Xs = np.append(training_Xs, X2_squares, 1)

param_vector = [0.1, 0.1, 0.1, 0.1, 0.1]
param_vector = np.transpose(param_vector)

loss = op.loss_function(param_vector, training_Xs, training_Y)
print(loss)
teta_nonlinear = op.derivative(training_Xs, training_Y)
print(teta_nonlinear)
loss = op.loss_function(teta_nonlinear, training_Xs, training_Y)
print(loss)



