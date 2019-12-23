import numpy as np 


#User-based 
class CollaborativeFiltering:
    '''
    Arributes:
        X: input ratings including the active user
        active: the index of the active user
    '''
    def __init__(self, X, active=0):
        self.X = X
        self.N, self.d = X.shape
        self.active = active
        self.activeUser = X[active]

    def rateItems(self):  
        #remove the active user's ratings 
        ratings = np.vstack((self.X[0:self.active,:], self.X[self.active+1:self.N,:]))
        #find out the items needed to rate for the active user
        items_to_rate = np.nonzero(np.isnan(self.activeUser))
        #get other users's ratings on the items to rate
        ratings_subset = ratings.T[items_to_rate]
        if len(ratings_subset) == 0:
            return

        similarity = self._calculateUserSimilarity(ratings)
        #weight the other users's ratings by multipling the similarity
        weighted_ratings = ratings_subset * similarity
        
        predictRatings = np.zeros(weighted_ratings.shape[0])
        for i in range(len(predictRatings)):
            #find the non-Nan weighted rating
            index = np.nonzero(~np.isnan(weighted_ratings[i]))
            if len(weighted_ratings[i][index]) != 0 and np.sum(similarity[index]) != 0:
                predictRatings[i] = np.sum(weighted_ratings[i][index])/np.sum(similarity[index])
        return predictRatings

    #using cos as the metric of similarity
    def _calculateUserSimilarity(self, ratings):
        #initialize the similarity as [0,0,...]
        similarity = np.zeros(ratings.shape[0])
        for i in range(len(similarity)):
            #find the common non-Nan features of the active user' ratings and i-th user's ratings, use them to compute the similarity
            index = np.nonzero(~np.isnan(self.activeUser) & ~np.isnan(ratings[i]))
            if len(ratings[i][index]) != 0 and len(self.activeUser[index]) != 0 and not np.all(ratings[i][index] == 0) and not np.all(self.activeUser[index] == 0):
                similarity[i] = ratings[i][index].dot(self.activeUser[index])/np.sqrt(np.sum(ratings[i][index]**2)*np.sum(self.activeUser[index]**2))

        return similarity


if __name__ == '__main__':
    X = np.array([[9,6,8,4,np.nan],[2,10,6,np.nan,8],[5,9,np.nan,10,7],[np.nan,10,7,8,np.nan]])
    cf = CollaborativeFiltering(X, 3)
    print(cf.rateItems())
    



