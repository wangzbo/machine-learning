import numpy as np 
import numpy.linalg as lg

class LinearRegressor:

    def __init__(self, X, y):
        self.X = X
        self.y = y

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


if __name__ == '__main__':
    l = LinearRegressor([[1,0],[0,-1],[1,-2],[0,0.5],[-2,0],[-1,4]],[2,2.9,9.9,-3.2,-7.1,-20.5])
    print(l.ridge(5.0))
        