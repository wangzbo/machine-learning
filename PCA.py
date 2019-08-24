import numpy as np 

def PCA(X, outputD):
    N,d = X.shape
    if outputD > d:
        outputD == d
    eig_value, eig_vector = np.linalg.eig(X.T.dot(X))
    #print(eig_value)
    #print(eig_vector)
    sorted_eig_value = np.argsort(eig_value)[::-1]
    top_eig_vector = eig_vector[:,sorted_eig_value[:outputD]]
    return top_eig_vector.T

def tansform(X, outputD):
    X = np.array(X)
    X -= np.mean(X, axis = 0)
    top_eig_vec = PCA(X, outputD)
    return np.dot(top_eig_vec, X.T).T

if __name__ == '__main__':
    X = np.array([[-1, -1, 2, 3], [-2, -1, 1, -3], [-3, -2, 1, -1], [1, 1, 2, 3], [2, 1, -2, 3], [3, 2, -1, 2]], dtype=np.float64)
    print(tansform(X, 2))


