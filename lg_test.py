print(__doc__)


# Code source: Jaques Grobler
# License: BSD 3 clause


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

import linear_model as us


plt.title("Toy Dataset")
X1,Y1 = make_regression(n_samples=100, n_features=1, n_informative=1, n_targets=1)

X1_train = X1[:-20]
Y1_train = Y1[:-20]

X1_test = X1[-20:]
Y1_test = Y1[-20:]

'''regr = linear_model.LinearRegression()
regr.fit(X1_train, Y1_train)

# Make predictions using the testing set
Y1_pred = regr.predict(X1_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(Y1_test, Y1_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(Y1_test, Y1_pred))

plt.scatter(X1, Y1, marker='o', s=25, edgecolor='k')
plt.plot(X1_test, Y1_pred, color='blue', linewidth=3)
plt.show()
'''


our_rg = us.LinearRegression()
our_rg.fit(X1_train, Y1_train)
our_Y1_pred = our_rg.predict(X1_test)
