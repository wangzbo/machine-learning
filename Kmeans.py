import numpy as np
import random

class Kmeans:

    def __init__(self, X, k):
        self.X = X
        self.k = k
        self.centerPoints = None #np.array([])
        self.clusters = None

    def _initializeCenterPoints(self, N):
        return random.sample(range(N), self.k)

    def _calDistance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))

    def _getClusters(self, data):
        clustered_data = data[self.clusters[0]]
        for k in range(1,self.k):
            clustered_data = np.vstack((clustered_data, data[self.clusters[k]]))
        return self.centerPoints, clustered_data, np.array([len(self.clusters[k]) for k in range(self.k)])

    def cluster(self, T = 100):
        data = np.array(self.X)
        N, d = data.shape
        self.centerPoints = data[self._initializeCenterPoints(N)]
        for i in range(T):
            self.clusters = [list() for i in range(self.k)]
            for j in range(N):
                min_dis = float('inf')
                cluster_index = 0
                for k in range(self.k):
                    dis = self._calDistance(data[j],self.centerPoints[k])
                    if dis < min_dis:
                        min_dis = dis
                        cluster_index = k 
                self.clusters[cluster_index].append(j)
            self.centerPoints = np.array([np.mean(data[self.clusters[k]], axis=0) for k in range(self.k)])    

        return self._getClusters(data)

if __name__ == '__main__':

    X = [[0.3,1.1],[0.8,1.4],[0.5,1.8],[0.2,1.5],[0.4,1],[-1.2,-1.5],[-1.7,-1.3],[-1.9,-1],[-1.6,-1.6],[-1.2,-1.8],[3,-1],[3.5,-1.6],[3.7,-1.8],[3.2,-1.1],[4,-2]]
    km = Kmeans(X,3)
    means, clustered_data, clustered_num = km.cluster(10)
    print(means)
    print(clustered_data)
    print(clustered_num)