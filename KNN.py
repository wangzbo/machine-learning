import numpy as np

class KNN:
    def __init__(self, X, y, k):
        self.X = X
        self.y = y
        self.neighbors = k

    def __partition(self, distances, labels, start, end):
        if start < 0 or end > distances.size or start >= end:
            return 0
        low = start 
        high = end - 1
        dis = distances[start] 
        label = labels[start]
        while low < high:
            while low < high and distances[high] >= dis:
                high -= 1
            distances[low] = distances[high]
            labels[low] = labels[high]
            while low < high and distances[low] <= dis:
                low += 1
            distances[high] = distances[low]
            labels[high] = labels[low]

        distances[low] = dis
        labels[low] = label
        return low

    def __getKNeighbors(self, distances, labels, start, end):
        partition = self.__partition(distances, labels, start, end)
        if partition == self.neighbors - 1:
            return
        elif partition > self.neighbors - 1:
            self.__getKNeighbors(distances, labels, start, partition)
        else:
            self.__getKNeighbors(distances, labels, partition+1, end)


    def knnClassifier(self, x):
        train_data, train_label = np.array(self.X), np.array(self.y)
        N, d = train_data.shape
        distances = np.array([np.sum((np.array(x)-train_data[i])*(np.array(x)-train_data[i])) for i in range(N)])
        self.__getKNeighbors(distances, train_label, 0, N)

        k_labels = train_label[:self.neighbors]
        print(k_labels)
        uniqueLabels,indices = np.unique(k_labels,return_counts = True)
        majority = [uniqueLabels[i] for i in range(len(uniqueLabels)) if indices[i] == np.max(indices)]

        print(majority)
        return majority[0]

    def knnRegression(self, x):
        train_data, train_label = np.array(self.X), np.array(self.y)
        N, d = train_data.shape
        distances = np.array([np.sum((np.array(x)-train_data[i])*(np.array(x)-train_data[i])) for i in range(N)])
        self.__getKNeighbors(distances, train_label, 0, N)

        k_labels = train_label[:self.neighbors]
        print(k_labels)
        return np.sum(k_labels)/self.neighbors


if __name__ == '__main__':
    k = KNN([[0,3],[1,3],[2,3],[3,3],[3,2],[3,1],[3,0],[2,0],[1,0],[0,0],[0,1],[0,2],[1,2],[2,2],[2,1],[1,1]],[1,1,1,1,-1,-1,-1,-1,-1,-1,1,1,1,1,-1,-1],4)
    k1 = KNN([[0,3],[1,3],[2,3],[3,3],[3,2],[3,1],[3,0],[2,0],[1,0],[0,0],[0,1],[0,2],[1,2],[2,2],[2,1],[1,1]],[1,2,3,4,5,6,7,8,7,6,5,4,3,2,1,0],4)
    print(k.knnClassifier([0.8,1.3]))
    print(k1.knnRegression([2.1,2.8]))