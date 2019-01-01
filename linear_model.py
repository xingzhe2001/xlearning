#一个首先的线性回归, numpy不熟悉暂时用循环做了些工作

import numpy as np

class LinearRegression:
    def __init__(self, normalize = True, alpha=0.3):
        self.coef_ = 0.0
        self.intercept = 0.
        self.theta = None
        self.normalize = normalize
        self.offset = 1.0
        self.scalar = 1.0
        self.alpha = alpha
        self.iter_count = 0
        pass

    #for n feature, x = 0:n, theta = 0:n
    def __hypothetic(self, x):
        return np.dot(self.theta, x) + self.intercept

    #distance for single sample or all the samples
    def __error_dist(self, x, y):
        return self.__hypothetic(x) - y

    #cost function J of theta = \frac{1}{2m}\sum __error_dist^T __error_dist
    def __Jfunction(self):        
        sum = 0
        
        for i in range(0, self.m):
            err = self.__error_dist(self.x[i], self.y[i])
            sum += np.dot(err, err)
        
        return 1/(2 * self.m) * sum

    #\frac{1}{m}\sum X^T(\theta_TX-y)
    def __partialderiv_J_func(self):
        sum = 0

        for i in range(0, self.m):
            err = self.__error_dist(self.x[i], self.y[i])
            sum += np.dot(self.x[i], err)
   
        return 1/self.m * sum

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
        threshold = 0.01

        #repeat until convergence
        while threshold < cost:
            self.theta = self.theta - self.alpha * self.__partialderiv_J_func()
            self.intercept = self.intercept - self.alpha * self.__partialderiv_J_func_for_intersect()
            cost = self.__Jfunction()
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