import numpy as np 
import LinearRegressor

class MatrixFactorization:

    def __init__(self, R, d):
        self.R = np.array(R)
        self.d = d
        self.M, self.N = self.R.shape
        self.W = None
        self.V = None

    def _calError_in(self):
        error = [(self.R[i][j]-np.inner(self.W[i],self.V[j]))**2 for i in range(self.M) for j in range(self.N) if not np.isnan(self.R[i][j])]
        return np.sum(error)/len(error)
    
    def alternative_least_square(self):
        self.W = np.random.rand(self.M, self.d)
        self.V = np.random.rand(self.N, self.d)
        error_in = self._calError_in()
        error_in_old = 0
        while np.abs(error_in-error_in_old) > 0.000001:
            for i in range(self.M):
                v_indices = [j for j in range(self.N) if not np.isnan(self.R[i][j])]
                R_mn = self.R[i, v_indices]
                V_n = self.V[v_indices]
                self.W[i] = LinearRegressor.LinearRegressor(V_n, R_mn).ridge(0.1,False)
            for i in range(self.N):
                w_indices = [j for j in range(self.M) if not np.isnan(self.R[j][i])]
                R_mn = self.R[w_indices, i]
                W_m = self.W[w_indices]
                self.V[i] = LinearRegressor.LinearRegressor(W_m, R_mn).ridge(0.2,False)          
            error_in_old = error_in
            error_in = self._calError_in()
        '''
        print(error_in)
        print(self.W)
        print(self.V)
        print(np.mean(self.W))
        print(np.mean(self.V))
        '''

    def SGD_matrix_factor(self, T, learning_rate = 0.001):
        self.W = np.random.rand(self.M, self.d)
        self.V = np.random.rand(self.N, self.d)
        for i in range(T):
            m, n = np.random.randint(self.M), np.random.randint(self.N)
            if np.isnan(self.R[m][n]):
                continue
            residual =  self.R[m][n]-np.inner(self.W[m],self.V[n])
            V_old = self.V[n].copy()
            self.V[n] = self.V[n] + learning_rate*residual*self.W[m]
            self.W[m] = self.W[m] + learning_rate*residual*V_old

    def predict(self,m,n):
        return np.inner(self.W[m],self.V[n])

if __name__ == '__main__':
    R = [[np.nan,8.2,4.5,np.nan,7.3,np.nan],
         [np.nan,np.nan,6.5,7.8,8.1,np.nan],
         [9.4,np.nan,7.3,6.6,np.nan,8.2],
         [5.9,8.8,np.nan,7.7,np.nan,8.8],
         [7.4,7.2,8.6,np.nan,np.nan,np.nan],
         [np.nan,np.nan,np.nan,6.3,7.4,8.5],
         [np.nan,9.1,5.7,7.2,np.nan,8.9],
         [6.9,7.4,np.nan,np.nan,7.7,7.9],
         [8.1,np.nan,5.6,7.8,8.3,np.nan],
         [np.nan,np.nan,7.5,np.nan,7.2,np.nan],
         [np.nan,6.6,np.nan,7.1,8.5,9.3],
         [7.6,np.nan,6.8,np.nan,6.9,8.7],
         [np.nan,np.nan,np.nan,7.6,7.8,8.1],
         [6.4,np.nan,6.7,7.5,np.nan,np.nan],
         [np.nan,8.2,7.1,np.nan,np.nan,7.7]]

    #the d should not be too big
    mf = MatrixFactorization(R, 3)
    mf.alternative_least_square()
    print(mf.predict(1,5))
    
    mf.SGD_matrix_factor(10000)
    print(mf.predict(1,5))
    