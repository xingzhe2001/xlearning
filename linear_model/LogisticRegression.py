#一个手写的逻辑回归

import numpy as np

class LogisticRegression:
    def __init__(self, normalize = True, alpha=0.3):
        self.coef_ = 0.0
        self.intercept = 0.
        self.theta = None
        self.normalize = normalize
        self.offset = 1.0
        self.scalar = 1.0
        self.alpha = alpha
        self.iter_count = 0
        self.cost_history=[]
        pass
    
    # e/(1+e^ax)
    def __sigmoid(self, z):        
        epart = np.exp( z )
        return epart / (1 + epart)

    #for n feature, x = 0:n, theta = 0:n
    def __hypothetic(self, x):
        z = np.dot(self.theta, x) + self.intercept
        return self.__sigmoid(z)

    #distance for single sample or all the samples
    def __error_dist(self, x, y):
        return self.__hypothetic(x) - y

    #y log h + (1 - y) log( 1- h )
    def __loglikelihood(self, x, y):
        h = self.__hypothetic(x)
        return y * np.log(h) + (1 - y)*np.log(1 - h)

    #J = - mean of likelihood
    def __Jfunction(self):        
        sum = 0
        
        for i in range(0, self.m):
            sum += self.__loglikelihood(self.x[i], self.y[i])
        
        return 1/self.m * sum

    #mean of ( h - y ) * x_j = [dist matrix for sample] dot sample
    def __partialderiv_J_func(self,):

        h = np.zeros(self.m)        
        for i in range(0, self.m):
            h[i] = self.__hypothetic(self.x[i])

        dist = h - self.y
        
        return np.asarray(np.mat(dist.T) * self.x) / self.m

        #\frac{1}{m}\sum X^T(\theta_TX-y)
    def __partialderiv_J_func_for_intersect(self):
        sum = 0

        for i in range(0, self.m):
            err = self.__error_dist(self.x[i], self.y[i])
            sum += err
   
        return 1/self.m * sum

    #\theta = \theta - \alpha * \partial costfunction
    def __gradient_descent(self):
        cost = 100000.0
        last_cost = 200000.0
        threshold = 0.01

        self.iter_count = 0
        #repeat until convergence
        while abs(cost - last_cost) > 0.0001:
            last_cost = cost
            self.theta = self.theta - self.alpha * self.__partialderiv_J_func()
            self.intercept = self.intercept - self.alpha * self.__partialderiv_J_func_for_intersect()
            cost = -self.__Jfunction()
            self.cost_history.append(cost)
            print('iter=%d deltaCost=%f'%(self.iter_count, last_cost - cost))
            self.iter_count += 1


    def __calculate_norm_params(self, x):
        offset = np.zeros(self.n_feature)
        scalar = np.ones(self.n_feature)
        for feature_idx in range(0, self.n_feature):
            col = x[:, np.newaxis, feature_idx]
            min = col.min()
            max = col.max()
            mean = col.mean()

            if( min != max):
                scalar[feature_idx] = 1.0/(max - min)
            else:
                scalar[feature_idx] = 1.0/max
            
            offset[feature_idx] = mean

        return offset, scalar

    def __normalize(self, x):
        return (x - self.offset) * self.scalar       

    
    def fit(self, x, y):
        if x.shape[0] != y.shape[0]:
            raise 'x, y have different length!'
     
        self.m = x.shape[0]
        self.n_feature = x.shape[1]
        self.theta = np.zeros(x[0].size)

        if self.normalize:
            self.offset, self.scalar = self.__calculate_norm_params(x)
            self.x = self.__normalize(x)
        else:
            self.x = x

        self.y = y 
        
        self.__gradient_descent()

        self.coef_ = self.theta
        pass

    def predict(self, x):
        y_pred = []
        for element in x:
            xi = element
            if self.normalize:
                xi = self.__normalize(element)

            y_pred.append(self.__hypothetic(xi))

        return y_pred

#Test
'''
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.datasets import make_classification

X1,Y1 = make_classification(n_samples = 100, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1)

#plt.show()

lr = LogisticRegression(normalize=False)
lr.fit(X1, Y1)
print(lr)

from sklearn.linear_model import LogisticRegression as skLR

ref_lr = skLR()
ref_lr.fit(X1, Y1)

def __sigmoid(z):        
    epart = np.exp( z )
    return epart / (1 + epart)

z = np.arange(-20,20)
plt.plot(z, __sigmoid(z))
plt.title('sigmoid')
plt.show()

x_axis = np.arange(-5,5)
plt.scatter(X1[:,0], X1[:,1], marker='o', c=Y1, s=25, edgecolor='k')
y_curve = -(x_axis * lr.theta[0][0] + lr.intercept[0])/lr.theta[0][1]

plt.xlabel('x1')
plt.ylabel('x2')

plt.text(-4, 0, 'green:xlearning', color='g')
plt.text(-4, 1, 'red:sklearn', color='r')

plt.plot(x_axis, y_curve, 'g-')
plt.plot(x_axis,  -(x_axis * ref_lr.coef_[0][0] + ref_lr.intercept_[0])/ref_lr.coef_[0][1], 'r-')
plt.show()


plt.plot(np.arange(0, lr.iter_count), lr.cost_history)

plt.show()
'''