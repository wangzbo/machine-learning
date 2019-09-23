import numpy as np 
from scipy import optimize
import LogisticRegressor

class SVM:
    '''
    Arributes:
        X: input features of train data
        y: labels of train data
        kernel: kernel used, supporting linear, polynomial and gaussian
        theta: parameter in polynomial kernel
        gama: parameter in polynomial and gaussian kernel
        Q: parameter in polynomial kernel
        w: weight
        b: intercept
    '''
    def __init__(self, X, y, kernel = 'linear',theta = 1.0, gama = 1.0, Q = 2):
        self.X = X
        self.y = y
        self.kernel = kernel
        self.theta = theta
        self.gama = gama
        self.Q = Q
        self.w = None
        self.b = None


    def svm(self, C = 1.0):
        train_data, train_label = np.array(self.X), np.array(self.y)
        N, d = train_data.shape

        alpha = np.random.randn(N)
        Q = np.zeros((N, N),dtype=np.float)
        for i in range(N):
            for j in range(N):
                if self.kernel == 'linear':
                    Q[i][j] = np.inner(train_data[i], train_data[j])*train_label[i]*train_label[j]
                elif self.kernel == 'polynomial':
                    Q[i][j] = ((np.inner(train_data[i],train_data[j])*self.gama+self.theta)**self.Q)*train_label[i]*train_label[j]
                elif self.kernel == 'gaussian':
                    Q[i][j] = np.exp(-self.gama*np.sum((train_data[i]-train_data[j])**2))*train_label[i]*train_label[j]
        
        p = -1*np.ones(N)

        fun = lambda _alpha: 0.5*_alpha.dot(Q).dot(_alpha.T)+np.inner(p,_alpha)
        bound = tuple([(0,C)]*N)
        cons = ({'type': 'eq', 'fun': lambda _alpha: np.inner(_alpha, train_label)})

        optimal_alpha = optimize.minimize(fun, alpha, bounds = bound, constraints = cons)
        
        return optimal_alpha.x
        
    def predict_svm(self, alpha, x, C=1.0):
        train_data, train_label = np.array(self.X), np.array(self.y)
        N, d = train_data.shape
        if self.kernel == 'linear':
            w = (alpha * train_label).dot(train_data)
            #support vector index
            sv_alpha = [i for i in range(N) if np.abs(alpha[i]) > 0.000001 and np.abs(alpha[i]) < C]
            #print(sv_alpha)
            b = train_label[sv_alpha[0]] - np.inner(w, train_data[sv_alpha[0]])
            return np.sign(np.inner(w,x)+b)

        elif self.kernel == 'polynomial':
            return np.sign(np.sum([alpha[i]*train_label[i]*((self.theta+self.gama*np.inner(self.X[i],x))**self.Q) for i in range(N)]))

        elif self.kernel == 'gaussian':
            return np.sign(np.sum([alpha[i]*train_label[i]*np.exp(-self.gama*np.sum((self.X[i]-np.array(x))**2)) for i in range(N)]))
    
    def probabilistic_svm(self, C = 1.0, iteration = 1000):
        self.svm(C)
        transform_features = np.array([np.inner(self.w,x)+self.b for x in self.X]).reshape(len(self.X),1)
        l = LogisticRegressor.LogisticRegressor(transform_features,self.y)
        print(l.logisticRegression(iteration))
        return l.logisticRegression()

    def svr(self, C = 1.0, epsilon = 1.0):
        train_data, train_label = np.array(self.X), np.array(self.y)
        N, d = train_data.shape

        alpha_pos = np.random.randn(N)
        alpha_neg = np.random.randn(N)
        alpha =  np.hstack((alpha_pos,alpha_neg))
        Q = np.zeros((N, N),dtype=np.float)
        for i in range(N):
            for j in range(N):
                if self.kernel == 'linear':
                    Q[i][j] = np.inner(train_data[i], train_data[j])
                elif self.kernel == 'polynomial':
                    Q[i][j] = (np.inner(train_data[i],train_data[j])*self.gama+self.theta)**self.Q
                elif self.kernel == 'gaussian':
                    Q[i][j] = np.exp(-self.gama*np.sum((train_data[i]-train_data[j])**2))
        
        fun = lambda _alpha: 0.5*(_alpha[:N]-_alpha[N:]).dot(Q).dot((_alpha[:N]-_alpha[N:]).T)+np.inner(epsilon-train_label,_alpha[:N])+np.inner(epsilon+train_label,_alpha[N:])
        bound = tuple([(0,C)]*2*N)
        cons = ({'type': 'eq', 'fun': lambda _alpha: np.sum(_alpha[:N])-np.sum(_alpha[N:])})

        optimal_alpha = optimize.minimize(fun, alpha, bounds = bound, constraints = cons)
        
        return optimal_alpha.x 


    def predict_svr(self, alpha, x, C = 1.0):
        train_data, train_label = np.array(self.X), np.array(self.y)
        N, d = train_data.shape
        alpha_pos = alpha[:N]
        alpha_neg = alpha[N:]
        if self.kernel == 'linear':
            w = (alpha_pos - alpha_neg).dot(train_data)
            #support vector index
            sv_alpha = [i for i in range(N) if np.abs(alpha_pos[i]) > 0.000001 and np.abs(alpha_pos[i]) < C]
            if len(sv_alpha) == 0:
                sv_alpha = [i for i in range(N) if np.abs(alpha_neg[i]) > 0.000001 and np.abs(alpha_neg[i]) < C]
            #print(sv_alpha)
            b = train_label[sv_alpha[0]] - np.inner(w, train_data[sv_alpha[0]])
            return np.inner(w,x)+b

        elif self.kernel == 'polynomial':
            beta = alpha_pos - alpha_neg
            return np.sum([beta[i]*((self.theta+self.gama*np.inner(self.X[i],x))**self.Q) for i in range(N)])

        elif self.kernel == 'gaussian':
            beta = alpha_pos - alpha_neg
            return np.sum([beta[i]*np.exp(-self.gama*np.sum((self.X[i]-np.array(x))**2)) for i in range(N)])

if __name__ == '__main__':
    s = SVM([[0,0],[-1,0],[1,-2],[3,0],[2,-1],[2,-2],[1,3],[-1,-1],[-1,1],[4,1]],[-1,-1,1,1,1,1,-1,-1,-1,1])
    #s.probabilistic_svm(1)
    alpha = s.svm()
    print(s.predict_svm(alpha,[-2,2]))

    s = SVM([[-4],[-3],[-2],[-1],[0],[1],[2],[3],[4],[5]],[11,5,1,-1,-1,1,5,11,19,29],'polynomial')
    alpha = s.svr(1,0.2)
    print(s.predict_svr(alpha,[-10]))

    
