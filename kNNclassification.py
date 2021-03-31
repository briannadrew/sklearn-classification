# kNNclassification.py

# Brianna Drew
# March 31, 2021
# ID: #0622446
# Lab #10, Part #3

# import required libraries and modules
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

x, y = sklearn.datasets.load_iris(return_X_y = True) # load iris dataset, returning data to x and class labels to y
x_trainingdata, x_testdata, y_trainingdata, y_testdata = train_test_split(x, y, test_size = 0.33) # split training and test data 2/3 to 1/3 respectively

iriskNN = KNeighborsClassifier(n_neighbors = 3) # create kNN classifier
iriskNN.fit(x_trainingdata, y_trainingdata) # apply kNN classifier to iris training data
prediction = iriskNN.predict(x_testdata) # get classification predictions resulting from the kNN classifier

print(classification_report(y_testdata, prediction)) # print classification report resulting from the kNN classifier
print(confusion_matrix(y_testdata, prediction)) # print confusion matrix resulting from the kNN classifier