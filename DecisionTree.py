import numpy as np 

class DecisionTreeNode:

    def __init__(self,value=0,feature_index=0,theta=0.0,left=None,right=None):
        self.value = value
        self.feature_index = feature_index
        self.theta = theta
        self.left = left
        self.right = right

class DecisionTree:

    #CART algorithm
    def __init__(self, X, y, max_depth = 0, classification = True):
        self.X = X
        self.y = y
        self.max_depth = max_depth
        if classification:
            self.root = self.buildDT_classification(np.array(X),np.array(y),1)
        else:
            self.root = self.buildDT_regression(np.array(X),np.array(y),1)

    def _preOrder(self, node):
        if node != None:
            print(node.value)
            self._preOrder(node.left)
            self._preOrder(node.right)

    def _terminated(self, data, label, depth):
        if depth == self.max_depth:
            return True
        N, d = data.shape
        data_flag = [(data[0] == d).all() for d in data] == [True]*N
        label_flag = [label[0] == l for l in label] == [True]*N
        return data_flag or label_flag
    
    def _impurity_classification(self, data, label):
        N, d = data.shape
        #Gini index
        u_label = np.unique(label)
        return 1-np.sum([(np.sum([l == ul for l in label])/(1.0*N))**2 for ul in u_label])

    def _impurity_regression(self, data, label):
        N, d = data.shape
        #square error
        return np.sum([(l-np.mean(label))**2 for l in label])/(1.0*N)

    def _branch_criteria(self, data, label, classification=True):
        N, d = data.shape
        w_impurity, feature_index, threshold, data1, label1, data2, label2 = float('inf'),0,0.0,None,None,None,None

        for i in range(d):
            feature = data[:,i]
            sorted_feature, sorted_label_index = np.sort(feature),np.argsort(feature)
        
            for j in range(N-1):
                if sorted_feature[j] == sorted_feature[j+1]:
                    continue
                theta = (sorted_feature[j]+sorted_feature[j+1])/2.0

                data_part1 = np.array([data[sorted_label_index[k]] for k in range(j+1)])
                label_part1 = np.array([label[sorted_label_index[k]] for k in range(j+1)])

                data_part2 = np.array([data[sorted_label_index[k]] for k in range(j+1,N)])
                label_part2 = np.array([label[sorted_label_index[k]] for k in range(j+1,N)])

                weighted_impurity = 0
                if classification:
                    weighted_impurity = self._impurity_classification(data_part1,label_part1) * data_part1.shape[0] 
                    + self._impurity_classification(data_part2,label_part2) * data_part2.shape[0]
                else:
                    weighted_impurity = self._impurity_regression(data_part1,label_part1) * data_part1.shape[0] 
                    + self._impurity_regression(data_part2,label_part2) * data_part2.shape[0]                
                
                if weighted_impurity < w_impurity:
                    w_impurity, feature_index, threshold, data1, label1, data2, label2 = weighted_impurity,i,theta,data_part1,label_part1,data_part2,label_part2
        
        return feature_index, threshold, data1, label1, data2, label2

    '''
    def _branch_function(self, feature_index, threshold, x):
        return (x[feature_index] <= threshold) + 1
    '''
     
    def constValue_classification(self, label):
        uniqueLabels,indices = np.unique(label,return_counts = True)
        majority = [uniqueLabels[i] for i in range(len(uniqueLabels)) if indices[i] == np.max(indices)]
        return majority[0]
        '''
        u_label = np.unique(label)
        majority_num, majority = 0, None
        for u in u_label:
            num = np.sum([l == u for l in label])
            if num > majority_num:
                majority_num = num
                majority = u
        return majority
        '''

    def constValue_regression(self, label):
        return np.mean(label)

    def buildDT_classification(self, data, label, depth):
        if self._terminated(data, label, depth):
            return DecisionTreeNode(self.constValue_classification(label))
        feature_index, theta, data1, label1, data2, label2 = self._branch_criteria(data,label)
        #print((feature_index, theta))
        return DecisionTreeNode(0,feature_index,theta,self.buildDT_classification(data1,label1,depth+1),self.buildDT_classification(data2,label2,depth+1))

    def buildDT_regression(self, data, label, depth):
        if self._terminated(data, label, depth):
            return DecisionTreeNode(self.constValue_regression(label))
        feature_index, theta, data1, label1, data2, label2 = self._branch_criteria(data,label,False)
        #print((feature_index, theta))
        return DecisionTreeNode(0,feature_index,theta,self.buildDT_regression(data1,label1,depth+1),self.buildDT_regression(data2,label2,depth+1))

    def predict(self, x):
        return self.DT_predict(self.root, x)

    def DT_predict(self, node, x):
        if node.left == None and node.right == None:
            return node.value
        if x[node.feature_index] <= node.theta:
            return self.DT_predict(node.left, x)
        else:
            return self.DT_predict(node.right, x)

if __name__ == '__main__':
    X = [[1,4],[1.5,3.5],[0.5,2.5],[0.5,1],[1.5,2],[2.5,3],[2.5,1.5],[2.5,0.5],[3.5,5],[3.5,3],[3.5,1.5],[4.5,2.5],[4.5,1.5],[5,3],[5,1]]
    y = [1,1,-1,-1,-1,1,1,-1,1,-1,-1,1,-1,1,-1]
    x = [2.2,0.8]
    dt = DecisionTree(X, y, 7)
    #dt._preOrder(dt.root)
    print(dt.predict(x))
    
