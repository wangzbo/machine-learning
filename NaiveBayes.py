import numpy as np

class NaiveBayes:

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.classes = np.unique(y)
        self.prob_classes = None
        self.prob_conditional = None

    def _get_prob_classes(self, N):
        prob_classes = [np.sum(np.equal(self.y,cl))/(1.0*N) for cl in self.classes]
        return dict(zip(self.classes,prob_classes))

    def _get_prob_conditional(self, N, d, alpha):
        prob_conditional = []
        for i in range(d):
            features = np.array(self.X)[:,i]
            u_features = np.unique(features)
            prob_feature = {}
            for cl in self.classes:
                class_num = np.sum(np.equal(self.y,cl))
                prob_feature[cl] = {}
                for ft in u_features:
                    feature_num = np.sum([np.equal(features[j],ft) for j in range(N) if self.y[j] == cl])
                    prob_feature[cl][ft] = (feature_num+alpha)/(1.0*(class_num+alpha*len(u_features)))
                
                prob_feature[cl]['unseen'] = alpha/(1.0*(class_num+alpha*len(u_features)))
            
            prob_conditional.append(prob_feature)
        
        return prob_conditional

    def _get_prob_conditional_gaussion(self, N , d):
        prob_conditional = []
        for i in range(d):
            features = np.array(self.X)[:,i]
            prob_feature = {}
            for cl in self.classes:       
                cls_features = [features[i] for i in range(N) if self.y[i] == cl]
                mean_variance = [np.mean(cls_features), np.var(cls_features)]
                prob_feature[cl] = mean_variance

            prob_conditional.append(prob_feature)
        
        return prob_conditional

    def multinomialBayes(self, alpha = 1.0):
        train_data, train_label = np.array(self.X), np.array(self.y)
        N, d = train_data.shape
        self.prob_classes = self._get_prob_classes(N)
        self.prob_conditional = self._get_prob_conditional(N, d, alpha)

    def GaussionBayes(self):
        train_data, train_label = np.array(self.X), np.array(self.y)
        N, d = train_data.shape
        self.prob_classes = self._get_prob_classes(N)
        self.prob_conditional = self._get_prob_conditional_gaussion(N, d)

    def predict(self, x):
        pred_prob = 0
        pred_label = self.classes[0]
        for cl in self.classes:
            pred_con = 1
            for i in range(len(x)):
                if x[i] in np.array(self.X)[:,i]:
                    pred_con *= self.prob_conditional[i][cl][x[i]]
                else:
                    pred_con *= self.prob_conditional[i][cl]['unseen']
            
            pred = self.prob_classes[cl] * pred_con
            if pred > pred_prob:
                pred_prob = pred
                pred_label = cl
        
        return pred_label

    def predict_gaussion(self, x):
        pred_prob = 0
        pred_label = self.classes[0]
        for cl in self.classes:
            pred_con = 1
            for i in range(len(x)):
                mean = self.prob_conditional[i][cl][0]
                var = self.prob_conditional[i][cl][1]
                pred_con = pred_con * 1.0 /(np.sqrt(2*np.pi*var))*np.exp(-(x[i]-mean)**2/(2*var))
            
            pred = self.prob_classes[cl] * pred_con
            if pred > pred_prob:
                pred_prob = pred
                pred_label = cl
        
        return pred_label


if __name__ == '__main__':
    n = NaiveBayes([[1,0,0,0],[0,1,0,1],[1,1,0,1],[0,1,1,1],[1,0,0,1],[0,0,0,0],[1,1,1,0],[0,1,1,1],[1,1,1,1],[0,0,1,1],[1,1,0,0],[1,1,0,0]],[0,0,1,1,0,0,1,1,1,1,0,0])
    n.multinomialBayes()
    print(n.predict([1,0,1,1]))
    print(n.predict([0,1,1,0]))
    print(n.predict([0,0,0,0]))
    print(n.predict([1,0,1,0]))
    print(n.predict([0,0,1,0]))
    print(n.predict([0,1,0,0]))
    print(n.predict([0,0,0,1]))

    from sklearn import datasets
    iris = datasets.load_iris()
    g = NaiveBayes(iris.data, iris.target)
    g.GaussionBayes()
    print(g.predict_gaussion(iris.data[0]))
    print(g.predict_gaussion(iris.data[22]))
    print(g.predict_gaussion(iris.data[44]))
    print(g.predict_gaussion(iris.data[66]))
    print(g.predict_gaussion(iris.data[88]))
    print(g.predict_gaussion(iris.data[110]))
    print(g.predict_gaussion(iris.data[132]))
    print(g.predict_gaussion([6,4,6,2]))
