import numpy as np 
import numpy.linalg as lg

class LinearRegressor:

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.alpha = 1.0
        self.kernel = 'linear'
        self.theta = 1.0
        self.gama = 1.0
        self.Q = 2

    def linearRegression(self):
        train_data, train_label = np.array(self.X), np.array(self.y)
        N, d = train_data.shape
        train_data = np.hstack((np.ones((N,1)), train_data))

        w = lg.inv(train_data.T.dot(train_data)).dot(train_data.T).dot(train_label)

        return w

    def ridge(self, alpha = 1.0):
        train_data, train_label = np.array(self.X), np.array(self.y)
        N, d = train_data.shape
        train_data = np.hstack((np.ones((N,1)), train_data))

        identity = np.identity(d+1)
        w = lg.inv(train_data.T.dot(train_data)+alpha*identity).dot(train_data.T).dot(train_label)

        return w

    def kernel_ridge(self, alpha = 1.0, kernel = 'linear',theta = 1.0, gama = 1.0, Q = 2):
        train_data, train_label = np.array(self.X), np.array(self.y)
        N, d = train_data.shape
        self.alpha = alpha
        self.kernel = kernel
        self.theta = theta
        self.gama = gama
        self.Q = Q

        if kernel == 'linear':
            return self.ridge(alpha)
        elif kernel == 'polynomial':
            K = np.zeros((N, N),dtype=np.float)
            for i in range(N):
                for j in range(N):
                    K[i][j] = (np.inner(train_data[i], train_data[j])*gama+theta)**Q
            
            beta = lg.inv(K+alpha*np.identity(N)).dot(train_label)
            return beta
        elif kernel == 'gaussion':
            K = np.zeros((N, N),dtype=np.float)
            for i in range(N):
                for j in range(N):
                    K[i][j] = np.exp(-gama*np.sum((train_data[i]-train_data[j])**2))
            
            beta = lg.inv(K+alpha*np.identity(N)).dot(train_label)
            return beta

    def predict(self, beta, x):
        if self.kernel == 'linear':
            return np.inner(beta,np.array([1]+x))
        elif self.kernel == 'polynomial':
            return np.sum([beta[i]*((self.theta+self.gama*np.inner(self.X[i],x))**self.Q) for i in range(len(beta))])
        elif self.kernel == 'gaussion':
            return np.sum([beta[i]*np.exp(-self.gama*np.sum((self.X[i]-np.array(x))**2)) for i in range(len(beta))])

if __name__ == '__main__':
    l = LinearRegressor([[1,0],[0,-1],[1,-2],[0,0.5],[-2,0],[-1,4]],[2,2.9,9.9,-3.2,-7.1,-20.5])
    print(l.ridge(1.0))
    #print(l.kernel_ridge(1.0,'linear'))
    #print(l.kernel_ridge(1.0,'polynomial'))
    beta = l.kernel_ridge(1.0,'polynomial')
    print(beta)
    print(l.predict(beta,[0.8,0.1]))
        
