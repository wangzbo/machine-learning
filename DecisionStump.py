import numpy as np 

def DecisionStump(X, y):
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
                err = np.sum([pred_label[k]!=label[sorted_label_index[k]] for k in range(N)])
                if err < error_num:
                    error_num, feature_index, direction, threshold = err,i,s,theta
        
    return direction, feature_index, threshold

def predict(s,i,theta,x):
    return s*np.sign(x[i]-theta)

X = [[74],[63],[28],[91],[15],[57],[57],[88],[32],[32],[81]]
y = [1,1,-1,1,-1,-1,-1,1,-1,-1,1]
s,i,theta = DecisionStump(X,y)
para = (s,i,theta)
print(para)
print(predict(s,i,theta,[70]))
print(predict(s,i,theta,[55]))







    



    

