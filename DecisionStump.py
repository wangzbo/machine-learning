import numpy as np 

#u is the weight for each data
def DecisionStump(X, y, u):
    data, label = np.array(X), np.array(y)
    N, d = data.shape
    error_num, feature_index, direction, threshold = N,0,1,0

    for i in range(d):
        feature = data[:,i]
        sorted_feature, sorted_label_index = np.sort(feature),np.argsort(feature)
        
        for s in [-1, 1]:
            for j in range(N):
                theta = 0
                if j == N-1:
                    theta = sorted_feature[j] + 1
                else:
                    if sorted_feature[j] == sorted_feature[j+1]:
                        continue
                    theta = (sorted_feature[j]+sorted_feature[j+1])/2.0
                pred_label = np.array([-s]*(j+1)+[s]*(N-j-1))
                err = np.sum([(pred_label[k]!=label[sorted_label_index[k]])*u[sorted_label_index[k]]/(1.0*np.sum(u)) for k in range(N)])
                
                if err < error_num:
                    error_num, feature_index, direction, threshold = err,i,s,theta
        
    return error_num, direction, feature_index, threshold

def predict(s,i,theta,x):
    return s*np.sign(x[i]-theta)

if __name__ == '__main__':
    '''
    X = [[74],[63],[28],[91],[15],[57],[57],[88],[32],[32],[81]]
    y = [1,1,-1,1,-1,-1,-1,1,-1,-1,1]
    u = [0.2,0.8,1.2,1.5,2,1.1,0.7,0.4,2.4,1,0.9]
    err,s,i,theta = DecisionStump(X,y,u)
    para = (err,s,i,theta)
    print(para)
    print(predict(s,i,theta,[70]))
    print(predict(s,i,theta,[55]))
    '''

    X = [[1,4],[1.5,3.5],[0.5,2.5],[0.5,1],[1.5,2],[2.5,3],[2.5,1.5],[2.5,0.5],[3.5,5],[3.5,3],[3.5,1.5],[4.5,2.5],[4.5,1.5],[5,3],[5,1]]
    y = [1,1,-1,-1,-1,1,1,-1,1,-1,-1,1,-1,1,-1]
    u = [0.03333333,0.03333333,0.13333333,0.03333333,0.03333333,0.03333333,0.13333333,0.03333333,0.03333333,0.13333333,0.03333333,0.03333333,0.03333333,0.03333333,0.03333333]
    err,s,i,theta = DecisionStump(X,y,u)
    para = (err,s,i,theta)
    print(para)
