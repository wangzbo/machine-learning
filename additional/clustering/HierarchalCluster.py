import numpy as np 

class HierarchalCluster:
    '''
    Attribute:
        X: data to be clustered
        n: how many clusters to form
    '''
    def __init__(self, X, n):
        self.X = X
        self.N, self.d = X.shape
        self.n = n
        self.clusters = list()
        self.distances = np.zeros([self.N,self.N], dtype=float)
        self._initialize()

    def _initialize(self):
        #initialize the clusters
        for i in range(self.N):
            cluster = np.empty([0,self.d])
            cluster = np.append(cluster, [self.X[i]], axis=0)
            self.clusters.append(cluster)
        #initialize the distance matrix
        for i in range(self.N - 1):
            for j in range(i+1, self.N):
                self.distances[i][j] = self.distances[j][i] = self._computeDistance(i, j)
    
    # update the distance matrix between the new cluster and other clusters
    def _updateDistance(self, loc):
        r, c = self.distances.shape
        for i in range(r):
            self.distances[i][loc] = self.distances[loc][i] = self._computeDistance(i, loc)
    
    #compute the distance of clusters, single-linkage, complete-linkage and group-average using Euclidean distance
    def _computeDistance(self, i, j, method = 'mean'):
        cluster_1, cluster_2 = self.clusters[i],self.clusters[j]
        distance = 0
        if method == 'center':
            centerPoint_1, centerPoint_2 = np.mean(cluster_1,axis=0),np.mean(cluster_2,axis=0)
            distance = np.sqrt(np.sum((centerPoint_1-centerPoint_2)**2))
        elif method == 'min':
            distance = np.sqrt(np.sum((cluster_1[0]-cluster_2[0])**2))
            for point_1 in cluster_1:
                for point_2 in cluster_2:
                    tempDis = np.sqrt(np.sum((point_1-point_2)**2))
                    if distance > tempDis:
                        distance = tempDis
        elif method == 'max':
            distance = np.sqrt(np.sum((cluster_1[0]-cluster_2[0])**2))
            for point_1 in cluster_1:
                for point_2 in cluster_2:
                    tempDis = np.sqrt(np.sum((point_1-point_2)**2))
                    if distance < tempDis:
                        distance = tempDis
        elif method == 'mean':
            tempDis = 0
            for point_1 in cluster_1:
                for point_2 in cluster_2:
                    tempDis += np.sqrt(np.sum((point_1-point_2)**2))
            distance = tempDis/(cluster_1.shape[0]*cluster_2.shape[0])

        return distance

    def _findShortestDistance(self):
        shortestDis = float("inf")
        loc_1 = loc_2 = 0
        r, c = self.distances.shape
        for i in range(r-1):
            for j in range(i+1, c):
                if self.distances[i][j] < shortestDis:
                    shortestDis = self.distances[i][j]
                    loc_1 = i
                    loc_2 = j

        return shortestDis, loc_1, loc_2


    def getClusters(self):
        while len(self.clusters) > self.n:
            #find the two clusters with shortest distance
            shortestDis, loc_1, loc_2 = self._findShortestDistance()
            #merge one cluster to the other cluster
            self.clusters[loc_1] = np.append(self.clusters[loc_1], self.clusters[loc_2], axis=0)
            #update the distance matrix for the new merged cluster
            self._updateDistance(loc_1)

            #delete the other cluster and its distances in the matrix
            self.clusters.pop(loc_2)
            self.distances = np.delete(self.distances, loc_2, axis=0)
            self.distances = np.delete(self.distances, loc_2, axis=1)
        
        return self.clusters


if __name__ == '__main__':
    X = np.array([[0.5, 0.3],[-0.3, 0.4],[0.2, -0.1],
                  [0, 2],[0.5, 1.8],[-0.3, 1.9],
                  [2, 0.4],[1.7, 0],[2.3, -0.2]])
    hc = HierarchalCluster(X, 3)
    clusters = hc.getClusters()
    print(clusters)


