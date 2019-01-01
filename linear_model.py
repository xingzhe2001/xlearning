import numpy as np

class LinearRegression:
    def __init__(self):
        self.coef_ = 0.0
        self.intercept = 0.
        self.theta = None
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
        alpha = 0.3


        #repeat until convergence
        while threshold < cost:
            self.theta = self.theta - alpha * self.__partialderiv_J_func()
            self.intercept = self.intercept - alpha * self.__partialderiv_J_func_for_intersect()
            cost = self.__Jfunction()



    
    def fit(self, x, y):
        if x.shape[0] != y.shape[0]:
            raise 'x, y have different length!'

        self.x = x
        self.y = y        
        self.m = x.shape[0]
        self.theta = np.zeros(x[0].size)

        self.__gradient_descent()

        self.coef_ = self.theta
        pass

    def predict(self, x):
        y_pred = []
        for element in x:
            y_pred.append(self.__hypothetic(element))

        return y_pred