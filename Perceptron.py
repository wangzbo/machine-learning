import numpy as np

class Perceptron:
    #X is train data with N*d, y is label data with N*1, y is in {-1, 1}
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def PLA(self):
        train_data, train_label = np.array(self.X), np.array(self.y)
        N, d = train_data.shape
        train_data = np.hstack((np.ones((N,1)), train_data))
        w = np.random.randn(d+1)
        i = start = 0

        while True:
            index = i % N
            if np.inner(w,train_data[index]) * train_label[index] < 0:
                w = w + train_data[index] * train_label[index]
                start = i
            if i == start + N:
                break 
            i += 1
        
        return w/np.sqrt(np.sum(np.square(w)))


if __name__ == '__main__':
    p = Perceptron([[1,1],[0,0],[1,-1],[0,1.5],[-2,1],[-1,5],[2,2.9],[-1,0.1],[3,4.1],[-4,-3.1]],[1,1,1,-1,-1,-1,1,-1,-1,1])
    print(p.PLA())