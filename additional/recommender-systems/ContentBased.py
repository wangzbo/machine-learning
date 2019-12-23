import numpy as np

'''
Parameters:
    X: input profiles of the items
    score: the score of the items
'''
def learnProfile(X, score):
    N, d = X.shape
    score = np.reshape(score, [N,1])
    profile = np.sum(X*score,axis=0)
    return profile/np.sum(profile)

'''
Parameters:
    profile: learned profiles 
    x: the profile of the item to be predicted
    fullScore: the full score for items
'''
def predict(profile,x,fullScore=10.0):
    return np.sum(profile*x)*fullScore


if __name__ == '__main__':
    X = np.array([[0,1,1,0],[1,1,1,1],[1,0,1,0]])
    score = np.array([2,10,8])
    profile = learnProfile(X, score)
    
    W = np.array([[1,1,0,1],[0,0,1,0],[1,0,1,0]])
    for i in range(W.shape[0]):
        print(predict(profile, W[i]))


