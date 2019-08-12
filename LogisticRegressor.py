import numpy as np

class LogisticRegressor:

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __sigmoid(self, s):
        return 1.0/(1+np.exp(-s))

    def logisticRegression(self, iteration = 100):
        train_data, train_label = np.array(self.X), np.array(self.y)
        N, d = train_data.shape
        train_data = np.hstack((np.ones((N,1)), train_data))
        w = np.random.randn(d+1)

        #gradient decent
        for i in range(iteration):
            gradient = 0
            for j in range(N):
                gradient = gradient + self.__sigmoid(-1*train_label[j]*np.inner(w,train_data[j]))*(-1)*train_label[j]*train_data[j]
                
            gradient /= (N*1.0)
            learn_rate = 0.01
            w = w - learn_rate*gradient

        return w/np.sqrt(np.sum(np.square(w)))

    def logisticRegressionReg(self, iteration = 100, C = 1.0):
        train_data, train_label = np.array(self.X), np.array(self.y)
        N, d = train_data.shape
        train_data = np.hstack((np.ones((N,1)), train_data))
        w = np.random.randn(d+1)

        #SGD
        for i in range(iteration):
            j = np.random.randint(0, N)
            gradient = self.__sigmoid(-1*train_label[j]*np.inner(w,train_data[j]))*(-1)*train_label[j]*train_data[j]+1.0/(N*C)*w
            learn_rate = 0.01
            w = w - learn_rate*gradient

        return w/np.sqrt(np.sum(np.square(w)))

        

if __name__ == '__main__':
    l = LogisticRegressor([[1,1],[0,0],[1,-1],[0,1.5],[-2,1],[-1,5],[2,2.9],[-1,0.1],[3,4.1],[-4,-3.1]],[1,1,1,-1,-1,-1,1,-1,-1,1])
    print(l.logisticRegressionReg(100000,10))

