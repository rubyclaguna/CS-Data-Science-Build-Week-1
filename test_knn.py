from sklearn import datasets 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.neighbors import KNeighborsClassifier
from knn import knn 

iris = datasets.load_iris() 

data = iris.data 
target = iris.target

# print(iris.data)
# print(iris.target) 
# print(iris.target_names)

# 0 : Setosa 
# 1 : Versicolor 
# 2 : Virginica 

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=0) 

# First Test: Sklearn KNN Classifier 
clf = KNeighborsClassifier(n_neighbors=5) 

clf.fit(X_train, y_train) 
pred = clf.predict(X_test) 

print("Sklearn Prediction:", pred) 

print("Accuracy Score:", accuracy_score(y_test, pred))

y_pred = clf.predict([X_test[0]]) 
print("y_pred:", y_pred)
 
print("-------------------------------------") 

# Second Test: My KNN model 
r_clf = knn(n_neighbors=5) 

r_clf.knn_fit(X_train, y_train) 
r_pred = r_clf.knn_predict(X_test) 

print("My Model's Prediction:", r_pred) 

print("Accuracy Score:", accuracy_score(y_test, r_pred))

y_predi = r_clf.knn_predict([X_test[0]]) 
print("y_pred:", y_pred) 

# Neighbors and euclidean distance
n = r_clf.get_neighbors(X_test[0]) 
print("Neighbors & Euclidean Distances:", n) 
