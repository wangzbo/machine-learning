import numpy as np 

class Point:
    '''
    Attribute:
        p: one data point to be clustered
        label: label of the date point, the value should be corePoint, borderPoint or outlier
        visited: whether the point is visited or not
        adjacent: the neighbour points of the point 
    '''
    def __init__(self, p, label='outlier', visited=False, adjacent=None):
        self.p = p
        self.label = label
        self.visited = visited
        self.adjacent = list()

class DBSCAN:
    '''
    Attribute:
        X: data points to be clustered
        radius: the radius of the point's neighbourhood
        minPoints: minimum point number included in the radius area.
    '''
    def __init__(self, X, radius, minPoints):
        self.X = X
        self.N, self.d = X.shape
        self.radius = radius
        self.minPoints = minPoints
        self.corePoints = np.array([],dtype=object)

        self.points = np.array([],dtype=object)
        for i in range(self.N):
            self.points = np.append(self.points, Point(self.X[i]))

    def clustering(self):

        #compute distances between points
        distances = np.zeros((self.N, self.N))
        for i in range(0,self.N-1):
            for j in range(i+1, self.N):
                distances[j][i] = distances[i][j] = self._calculateDistance(self.X[i],self.X[j])

        #label core points
        for k in range(self.N):
            adjacentNum = 0
            for l in range(self.N):
                if k!=l and distances[k][l] <= self.radius:
                    self.points[k].adjacent.append(self.points[l])
                    adjacentNum += 1
            
            if adjacentNum >= self.minPoints-1:
                self.corePoints = np.append(self.corePoints, self.points[k])
                self.points[k].label = 'corePoint'

        #label boreder points
        for p in self.points:
            if p.label != 'corePoint':
                for adj in p.adjacent:
                    if adj.label == 'corePoint':
                        p.label = 'borderPoint'
                        break
  
        clusters = list()
        for cp in self.corePoints:
            if not cp.visited:
                cluster = list()
                self._getCluster(cp, cluster)
                clusters.append(np.array(cluster))

        return clusters


    def _getCluster(self, corePoint, cluster):
        cluster.append(corePoint)
        corePoint.visited = True
        for adj in corePoint.adjacent:  
            if adj.label == 'corePoint' and not adj.visited:
                self._getCluster(adj, cluster)
            elif not adj.visited:
                cluster.append(adj)
                adj.visited = True


    def _calculateDistance(self, p1, p2):
        return np.sqrt((p1-p2).dot(p1-p2))

    
if __name__ == '__main__':

    #X = np.array([[0,2],[0,0],[0.5,2.2],[0.5,-0.3],[1,3],[1,-1],[1.5,4.2],[1.5,-2.2],[-0.5,-0.3],[-0.5,2.2],[-1,3],[-1,-1],[-1.5,4.2],[-1.5,-2.2],[2,5.8],[1.7,5]])
    X = np.array([[0,3.1],[1,2.8],[2,1.7],[3,0],[0,-2.9],[1,-2.7],[2,-1.8],[-1,2.8],[-2,1.7],[-3,0],[-2,-1.8],[-1,-2.8],[2.5, 1.65],[2.5,-1.66],[-2.5,1.65],[-2.5,-1.66],
                  [0,1],[1,0],[-1,0],[0,-1]])
    dbscan = DBSCAN(X, 1.8, 2)
    clusters = dbscan.clustering()
    for c in clusters:
        print([x.p for x in c])



