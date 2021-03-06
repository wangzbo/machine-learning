import numpy as np 

class NeuralNetwork:
    '''
    Attributes:
        X: input features of train data 
        N, d: dimention of X
        y: labels of train data
        hiddenlayers: how many hidden layers in NN
        neurals: how many neurals for each hidden layer
        outlDimension: dimension of the output layer
        W: weights of all layers
    '''
    
    def __init__(self, X, y, hiddenlayers, neurals, outlDimension):
        self.X = np.array(X)
        self.N, self.d = self.X.shape
        self.y = np.array(y)
        self.hiddenlayers = hiddenlayers
        self.neurals = neurals
        self.outlDimension = outlDimension
        self.W = list()

    def _initialize_weight(self):
        layers = [self.d]
        layers.extend(self.neurals)
        layers.extend([self.outlDimension])
        for i in range(len(layers)-1):
            self.W.append(np.random.uniform(-1,1,(layers[i], layers[i+1])))  

    def _initialize_gradient(self):
        W_grad = list()
        for l in range(len(self.W)):
            grad = np.zeros(self.W[l].shape)
            W_grad.append(grad)
        
        return W_grad

    def _shuffle(self):
        X_y = np.hstack((self.X,self.y))
        np.random.shuffle(X_y)
        self.X = X_y[:,:self.d]
        self.y = X_y[:,self.d:]

    def _compute_forward(self, x):
        X, S = list(),list()
        X.append(x)
        x_l = x
        for l in range(len(self.W)):
            s_l = np.dot(x_l,self.W[l])
            #the activation function of output layer is softmax, tanh for others
            if l == len(self.W)-1:
                x_l = np.exp(s_l)/np.sum(np.exp(s_l))
            else:
                x_l = np.tanh(s_l)
            S.append(s_l)
            X.append(x_l)

        return X, S

    def _compute_backward(self, S, s_l, label):
        DELTA = list()
        for l in range(len(self.W)):
            if l == 0:
                delta_l = self._compute_grad_last_layer(s_l,label)
            else:
                delta_l = np.dot(DELTA[l-1],self.W[len(self.W)-l].T)*(1-(np.tanh(S[len(self.W)-l-1]))**2)
            DELTA.append(delta_l)
            
        DELTA.reverse()
        return DELTA

    def _compute_grad_last_layer(self,s_l,label):
        #gradient of cross entropy error to the input of softmax layer
        y_hat_grad = np.array([np.exp(i)/np.sum(np.exp(i)) for i in s_l])-label
        return y_hat_grad

    def trainNN(self, epoch=200, batch_size=4, learning_rate=0.01, alpha=0.01):
        #using BP
        self._initialize_weight()
        #batch number and data indexes for all batches
        batches_num = int(self.N/batch_size if self.N%batch_size == 0 else self.N/batch_size+1)
        batches = np.array([np.array([i*batch_size+j for j in range(batch_size) if i*batch_size+j < self.N]) for i in range(batches_num)])
        for i in range(epoch):
            self._shuffle()
            for bc in range(batches_num):
                # for each batch
                train_data = self.X[batches[bc]]
                train_label = self.y[batches[bc]]
                W_grad = self._initialize_gradient()

                X, S = self._compute_forward(train_data)
                DELTA = self._compute_backward(S,S[-1],train_label)                  
                for l in range(len(self.W)):
                    for j in range(len(batches[bc])):
                        W_grad[l] += np.outer(X[l][j],DELTA[l][j])
                    W_grad[l] = W_grad[l]/len(batches[bc])
                    self.W[l] = self.W[l] - learning_rate*(W_grad[l]+alpha*self.W[l]/(1+self.W[l]**2)**2)

    def predict(self, x):
        _x = x
        for l in range(len(self.W)):
            _s = np.dot(_x, self.W[l])
            #the activation function of output layer is softmax, tanh for others
            if l == len(self.W)-1:
                _x = np.exp(_s)/np.sum(np.exp(_s))
            else:
                _x = np.tanh(_s)
                
        return _x

if __name__ == '__main__':

    X = [[1,2],[1,1],[3,2],[2,4],[4,3],[3,5],[4,1],[5,5],
         [6,2],[6,5],[5,1],[5,6],[8,3],[2,6],[7,4],[2,8],
         [-1,2],[-2,4],[-3,2],[-4,1],[-3,5],[-5,2],[-1,4],[-2,2],
         [-6,2],[-6,5],[-5,5],[-5,7],[-8,3],[-2,6],[-7,4],[-4,8],
         [-1,-1],[-3,-4],[-2,-3],[-5,-1],[-4,-5],[-3,-1],[-2,-8],[-6,-2],
         [-6,-5],[-5,-5],[-7,-5],[-5,-7],[-8,-3],[-2,-6],[-7,-4],[-1,-8],
         [2,-3],[3,-1],[1,-1],[4,-5],[6,-3],[2,-4],[5,-1],[2,-2],
         [6,-5],[5,-5],[7,-5],[5,-7],[8,-3],[3,-6],[7,-4],[1,-8]]

    y = [[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],
         [1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],
         [0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],
         [0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],
         [0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],
         [0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],
         [0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1],
         [0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1]]

    nn = NeuralNetwork(X,y,2,[3,5],4)
    nn.trainNN()
    #print(nn.W)
    print(nn.predict([2,2]))
    print(nn.predict([-6,3]))
    print(nn.predict([-5,-3]))
    print(nn.predict([3,-4]))
    print(nn.predict([-100,100]))
 
    
