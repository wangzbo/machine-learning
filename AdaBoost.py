import numpy as np 
import DecisionStump

class AdaBoost:

    def __init__(self, X, y, T = 100):
        self.X = X
        self.y = y
        self.T = T
        self.u = np.ones(len(y),dtype=np.float64)/(1.0*len(y))

    #using decision stump as basic learner
    def adaboosting(self):
        res = []
        for iter in range(self.T):
            err,s,i,theta = DecisionStump.DecisionStump(self.X,self.y,self.u)
            if err < 0.00000001:
                break
            update_para = np.sqrt((1.0-err)/err)
            for j in range(len(self.u)):
                if self.y[j] != s*np.sign(self.X[j][i]-theta):
                    self.u[j] *= update_para
                else:
                    self.u[j] /= update_para
            alpha = np.log(update_para)
            res.append([alpha,s,i,theta])
        
        return res

    def adaboost_classification(self, parameters, x):
        #print(parameters)
        score = np.sum([parameters[i][0]*parameters[i][1]*np.sign(x[parameters[i][2]]-parameters[i][3]) for i in range(len(parameters))])
        return np.sign(score)

if __name__ == '__main__':
    X = [[1,4],[1.5,3.5],[0.5,2.5],[0.5,1],[1.5,2],[2.5,3],[2.5,1.5],[2.5,0.5],[3.5,5],[3.5,3],[3.5,1.5],[4.5,2.5],[4.5,1.5],[5,3],[5,1]]
    y = [1,1,-1,-1,-1,1,1,-1,1,-1,-1,1,-1,1,-1]
    x = [2.2,0.7]
    ab = AdaBoost(X, y)
    paras = ab.adaboosting()
    print(ab.adaboost_classification(paras, x))

        



    