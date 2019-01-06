# Path hack.
import sys, os
sys.path.insert(0, os.path.abspath('..'))

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

from xlearning.linear_model import LinearRegression


X1,Y1 = make_regression(n_samples=100, n_features=10, n_informative=1, n_targets=1)

X1_train = X1[:-20]
Y1_train = Y1[:-20]

X1_test = X1[-20:]
Y1_test = Y1[-20:]

def test_fit():

      print('Test LinearRegression without normalize input, alpha = 0.3:\n')
      our_rg = LinearRegression(normalize=True, alpha=0.3)
      our_rg.fit(X1_train, Y1_train)
      our_Y1_pred = our_rg.predict(X1_test)

      # The performance
      print('Total iterate count: ', our_rg.iter_count)
      # The coefficients
      print('Coefficients: \n', our_rg.coef_)
      # The mean squared error
      print("Mean squared error: %.2f"
            % mean_squared_error(Y1_test, our_Y1_pred))
      # Explained variance score: 1 is perfect prediction
      print('Variance score: %.2f' % r2_score(Y1_test, our_Y1_pred))
      plt.plot(range(0, our_rg.iter_count), our_rg.cost_history)
      plt.show()

def test_params():
      iters = []
      iters_for_normalize = []

      errors = []
      errors_for_normalize = []

      alphas = np.arange(0.1, 1.1, 0.1)

      for alpha in alphas:
            our_rg = LinearRegression(normalize = False, alpha = alpha)
            our_rg.fit(X1_train, Y1_train)
            our_Y1_pred = our_rg.predict(X1_test)

            iters.append(our_rg.iter_count)
            errors.append(mean_squared_error(Y1_test, our_Y1_pred))

      for alpha in alphas:
            our_rg = LinearRegression(normalize = True, alpha = alpha)
            our_rg.fit(X1_train, Y1_train)
            our_Y1_pred = our_rg.predict(X1_test)

            iters_for_normalize.append(our_rg.iter_count)
            errors_for_normalize.append(mean_squared_error(Y1_test, our_Y1_pred))

      _, ax1 = plt.subplots()
      ax1.plot(alphas, iters, 'b-')
      ax1.plot(alphas, iters_for_normalize, 'g-')
      ax1.set_ylabel('iter_count', color = 'b')

      ax2 = ax1.twinx()
      ax2.plot(alphas, errors, 'r-')
      ax2.plot(alphas, errors_for_normalize, 'y-')
      ax2.set_ylabel('mean sqrt err', color = 'r')

      plt.show()

test_fit()
#test_params()


