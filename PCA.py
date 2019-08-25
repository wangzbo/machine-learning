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

def SVD(X, outputD):
    N,d = X.shape
    if outputD > d:
        outputD == d
    u,s,v_h = np.linalg.svd(X)
    return v_h[:outputD,:]

def tansform(X, outputD):
    X = np.array(X)
    X -= np.mean(X, axis = 0)
    top_eig_vec = PCA(X, outputD)
    #top_eig_vec = SVD(X, outputD)
    return np.dot(top_eig_vec, X.T).T

#Two classes classification
def LDA(X, y, outputD):
    X, y = np.array(X), np.array(y)
    N, d = X.shape
    if outputD > d:
        outputD == d
    X_0, X_1 = X[[i for i in range(N) if y[i] == 0]], X[[i for i in range(N) if y[i] == 1]]
    #compute Sw
    Sw = (len(X_0)-1)*(np.cov(X_0.T)+(len(X_1)-1)*np.cov(X_1.T))
    #print(Sw)
    #compute Sb
    u_0, u_1 = np.mean(X_0,axis=0), np.mean(X_1,axis=0)
    Sb = np.outer(u_0-u_1,u_0-u_1)
    #print(Sb)

    u,s,v_h = np.linalg.svd(np.linalg.inv(Sw).dot(Sb))
    return np.dot(v_h[:outputD,:], X.T).T

if __name__ == '__main__':
    X = np.array([[-1, -1, 2, 3], [-2, -1, 1, -3], [-3, -2, 1, -1], [1, 1, 2, 3], [2, 1, -2, 3], [3, 2, -1, 2]], dtype=np.float64)
    print(tansform(X, 2))

    X = np.array([[2,1], [2.1,1.2], [1.8,1.1], [4,3], [4.2,3.5], [4.4,2.8]], dtype=np.float64)
    y = np.array([0,0,0,1,1,1])
    print(LDA(X,y,1))

