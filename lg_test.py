import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

import linear_model as us


X1,Y1 = make_regression(n_samples=100, n_features=10, n_informative=1, n_targets=1)

X1_train = X1[:-20]
Y1_train = Y1[:-20]

X1_test = X1[-20:]
Y1_test = Y1[-20:]

our_rg = us.LinearRegression()
our_rg.fit(X1_train, Y1_train)
our_Y1_pred = our_rg.predict(X1_test)

# The coefficients
print('Coefficients: \n', our_rg.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(Y1_test, our_Y1_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(Y1_test, our_Y1_pred))
