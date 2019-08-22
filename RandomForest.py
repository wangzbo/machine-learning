import numpy as np 
import DecisionTree

class RandomForest:

    def __init__(self, X, y, n_trees = 100):
        self.X = X
        self.y = y
        self.n_trees = n_trees
        self.trees_root = np.array([],dtype= DecisionTree.DecisionTree)
        self.OOB_table = np.zeros([len(self.X),self.n_trees],dtype=np.int64)
        self.feature_importance = np.zeros([len(self.X[0])],dtype=np.float64)

    def _bootstrap(self, n):
        x_index = np.random.randint(len(self.X), size = n)
        return np.array(self.X)[x_index], np.array(self.y)[x_index], x_index

    def RF_classification(self, n):
        for i in range(self.n_trees):
            data, label, indices = self._bootstrap(n)
            self.OOB_table[indices,i] = 1
            root = DecisionTree.DecisionTree(data, label)
            self.trees_root = np.append(self.trees_root, root)

    def RF_OOB_error(self):
        error = 0
        total_num = len(self.X)
        for i in range(len(self.X)):
            preds = np.array([self.trees_root[j].predict(self.X[i]) for j in range(self.n_trees) if self.OOB_table[i][j] == 0])
            if len(preds) == 0:
                total_num -= 1
                continue
            pred_avg = np.sum(preds)/(1.0*len(preds))
            #using mean square error
            error += (self.y[i] - pred_avg)**2

        return error/(1.0*total_num)

    def RF_OOB_error_perm(self, feature_index):
        perm_pred_table = np.zeros([len(self.X),self.n_trees],dtype=np.int64)
        for t in range(self.n_trees):
            data_indices = [j for j in range(len(self.X)) if self.OOB_table[j][t] == 0]
            data_val = np.array(self.X)[data_indices]
            label_val = np.array(self.y)[data_indices]
            #perm on the feature_index to data_val
            np.random.shuffle(data_val[:,feature_index])
            for i in range(len(data_indices)):
                perm_pred_table[data_indices[i]][t] = self.trees_root[t].predict(data_val[i])

        error = 0
        total_num = len(self.X)
        for i in range(len(self.X)):
            preds = np.array([perm_pred_table[i][j] for j in range(self.n_trees) if self.OOB_table[i][j] == 0])
            if len(preds) == 0:
                total_num -= 1
                continue
            pred_avg = np.sum(preds)/(1.0*len(preds))
            #using mean square error
            error += (self.y[i] - pred_avg)**2

        return error/(1.0*total_num)

    def RF_feature_importance(self):
        for index in range(len(self.X[0])):
            self.feature_importance[index] = np.abs(self.RF_OOB_error()-self.RF_OOB_error_perm(index))
        self.feature_importance /= np.sum(self.feature_importance)
        return self.feature_importance   

    def RF_classification_predict(self, x):
        print([self.trees_root[i].predict(x) for i in range(self.n_trees)])
        print(np.sum([self.trees_root[i].predict(x) for i in range(self.n_trees)]))
        return np.sign(np.sum([self.trees_root[i].predict(x) for i in range(self.n_trees)]))#/self.n_trees


if __name__ == '__main__':
    X = [[1,4],[1.5,3.5],[0.5,2.5],[0.5,1],[1.5,2],[2.5,3],[2.5,1.5],[2.5,0.5],[3.5,5],[3.5,3],[3.5,1.5],[4.5,2.5],[4.5,1.5],[5,3],[5,1]]
    y = [1,1,-1,-1,-1,1,1,-1,1,-1,-1,1,-1,1,-1]
    x = [2.5,2]
    rf = RandomForest(X, y, 200)
    rf.RF_classification(8)
    print(rf.RF_feature_importance())
    print(rf.RF_classification_predict(x))
    
