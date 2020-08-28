# imports 

import numpy as np 
import scipy as sp 

class knn: 
    
    def __init__(self, n_neighbors=5):

        self.n_neighbors = n_neighbors 
    
    def knn_fit(self, X_train, y_train):
        """
        Fits the dataset into the model, X_train: training data, y_train: target values.  
        """
        self.X_train = X_train 
        self.y_train = y_train

    def euclidean_distance(self, a, b): 
        """
        Calculates the Euclidean Distance between two vectors. 
        The Euclidean Distance is the measurement of the distance(line) between two 
        points in Euclidean Space. 
        """
        distance = 0 

        for i in range(len(a)-1):
            distance += (a[i] - b[i]) **2 
        
        return np.sqrt(distance) 

    def knn_predict(self, X):
        """
        Makes predictions for X given the X_train and y_train data.  
        """
        predictions = []

        # Main loop iterating through len(X). 
        for i in range(len(X)):

            dist = []

            # For every row in X_train, finds the euclidean distance to X 
            # and appends values to dist list. 
            for row in self.X_train:

                distance = self.euclidean_distance(row, X[i])
                dist.append(distance)
         
            # Sorts dist in ascending order, specifies num of neighbors. 
            neighbors = np.array(dist).argsort()[:self.n_neighbors]

            # Counts class occurences in y_train. 
            count_neighbors = {}

            for n in neighbors:
                if self.y_train[n] in count_neighbors:
                    count_neighbors[self.y_train[n]] += 1
                else:
                    count_neighbors[self.y_train[n]] = 1

            predictions.append(max(count_neighbors,
                             key=count_neighbors.get))

        return predictions

    def get_neighbors(self, x):
        """ 
        Displays list of 5 neighbors and their euclidean distance. 
        """ 
        distances = [] 
        
        # For every row in X_train, finds the euclidean distance to X 
        # and appends values to distances list. 
        for i in self.X_train:
            dist = self.euclidean_distance(i, x)
            distances.append(dist) 
     
        neighbors = np.array(distances).argsort()[: self.n_neighbors] 

        n_val = []

        # Append the neighbors to their corresponding euclidean distance.
        for i in range(len(neighbors)): 
            n_index = neighbors[i] 
            e_dist = distances[i] 
            n_val.append((n_index, e_dist)) 

        return n_val