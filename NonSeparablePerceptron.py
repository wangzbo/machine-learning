import numpy as np 

class NonSeparablePerceptron:

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __calErrorNum(self, data, label, w):
        counter = 0
        for i in range(data.shape[0]):
            if np.inner(w,data[i]) * label[i] < 0:
                counter += 1

        return counter
    
    def pocket(self, iteration):
        train_data, train_label = np.array(self.X), np.array(self.y)
        N, d = train_data.shape
        train_data = np.hstack((np.ones((N,1)), train_data))
        w = np.random.randn(d+1)
        errorNum = N

        for i in range(iteration):
            index = np.random.randint(0, N)
            if np.inner(w,train_data[index]) * train_label[index] < 0:
                w_new = w + train_data[index] * train_label[index]
                if self.__calErrorNum(train_data, train_label, w_new) < errorNum:
                    w = w_new
                
        return w/np.sqrt(np.sum(np.square(w)))

if __name__ == '__main__':
    p = NonSeparablePerceptron([[1,1],[0,0],[1,-1],[0,1.5],[-2,1],[-1,5],[2,2.9],[-1,0.1],[3,4.1],[-4,-3.1]],[1,1,1,-1,-1,-1,1,-1,-1,1])
    print(p.pocket(1000))