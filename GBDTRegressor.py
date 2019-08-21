import numpy as np 
import DecisionTree
from sklearn.ensemble import GradientBoostingRegressor

class GBDTRegressor:

    def __init__(self, X, y, n_trees = 100, max_depth = 5):
        self.X = X
        self.y = y
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees_root = np.empty([0,2])

    def GBDTRegression(self):
        N, d = len(self.X), len(self.y)
        s = np.zeros(N, dtype=np.float64)
        for i in range(self.n_trees):
            #using pruned CART as base learner, mean square error 
            residual = np.array(self.y) - s
            root = DecisionTree.DecisionTree(self.X, residual, self.max_depth, False)
            g_x = np.array([root.predict(x) for x in self.X])
            alpha = np.inner(g_x,residual)/(1.0*np.inner(g_x, g_x)+0.00000001)
            s += alpha * g_x
            self.trees_root = np.append(self.trees_root, [[alpha, root]], axis=0)

    def GBDTRegression_predict(self,x):
        return np.sum([self.trees_root[i][0]*self.trees_root[i][1].predict(x) for i in range(self.n_trees)])

if __name__ == '__main__':
    # z = ln(x**2+y**2+1)
    X = [[-3,1],[-2,-3],[-1,-1],[0,0],[1,1],[1,3],[2,1],[2,2],[0,2],[3,3],[4,2],[4,-1],[3,-2],[-4,3],[-3,0],[5,0],[5,2],[-4,3],[-5,-3],[1,6],[6,2],[-3,6],[-7,4],[4,6],[8,1],[7,5],[-8,0],[-6,3],[9,1],[9,8]]
    #y = [2.4,2.64,1.1,0,1.1,2.4,1.79,2.2,1.61,2.94,3.04,2.89,2.64,3.26,2.3,3.26,3.4,3.26,3.56,3.64,3.71,3.83,4.19,3.97,4.19,4.32,4.17,3.83,4.42,4.98]
    y = [10,13,2,0,2,10,5,8,4,18,20,17,13,25,9,25,29,25,34,37,40,45,65,52,65,74,64,45,82,144]
    x = [10,10]
    gbdt = GBDTRegressor(X, y, 10, 2)
    gbdt.GBDTRegression()
    print(gbdt.GBDTRegression_predict(x))


    gbm0 = GradientBoostingRegressor()
    gbm0.fit(X,y)
    print(gbm0.predict([[10,10]]))