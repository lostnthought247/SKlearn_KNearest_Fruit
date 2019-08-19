#!/usr/bin/python3
""" An example SKLearn script that uses a structured data text file containing
information about various fruits to create a classifier tool to identify fruit
based on input features

Based on a training from Coursera course : Applied Machine Learning in Python
(University of Michigan) week 1"""

# import required python libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# injest text file data and create pandas dataframe
fruits = pd.read_csv('fruit_data_with_colors.txt', delim_whitespace=True)

# Create dictionary to make it easy to convert fruit labels to fruit names later
lookup_fruit_name = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))

# # ** Optional visualization **
# # Prints the first 10 rows of data and the "shape" of the fruits dataframe
# print(fruits.head())
# print(fruits.shape)

# Creates a "X" variable that includes all features, and a "Y" variable for
# the known classification labels in your training/test sets
X = fruits[["mass", "width", "height", "color_score"]]
y = fruits['fruit_label']

# Splits the data into training/test sets split intro 75%/25% with random seed
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# # ** Optional visualization **
# # Creates a "feature pair plot" to visualize relationships between data features
# from matplotlib import cm
# import matplotlib.pyplot as plt
# cmap = cm.get_cmap("gnuplot")
# scatter = pd.plotting.scatter_matrix(X_train, c=y_train, marker = "o", s=40, hist_kwds={"bins":15}, cmap=cmap)
# plt.show()

# # # ** Optional visualization **
# # Creates a 3d/rotatable graph of data based on width, height, and color_score
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection = "3d")
# ax.scatter(X_train["width"], X_train["height"], X_train["color_score"], c = y_train, marker = "o", s=100)
# ax.set_xlabel('width')
# ax.set_ylabel('height')
# ax.set_zlabel('color_score')
# plt.show()

# Create the ML Model using SKLearn K-Nearest Neighbor algorithm
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model using your "train" datasets
knn.fit(X_train, y_train)

# Test/Score the model using your "test" datasets
print(knn.score(X_test, y_test))

# # # ** Optional visualization **
# # Predict the classification for an individual piece of fruit given its traits
# fruit_prediction = knn.predict([[20, 4.3, 5.5, 0.70]])
# print(fruit_prediction)
#
# # use the fruit lookup dict to convert numerical result to fruit name string
# print("The prediction is:", lookup_fruit_name[fruit_prediction[0]])
#
# # Predict the classification for an individual piece of fruit given its traits
# fruit_prediction = knn.predict([[100, 6.3, 8.5, 0.70]])
# print(fruit_prediction)
#
# # use the fruit lookup dict to convert numerical result to fruit name string
# print("The prediction is:", lookup_fruit_name[fruit_prediction[0]])


# ** Optional visualization **
# Loops through models with K values 1-20 and test/displays accuracy of each
k_range = range(1,20)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))

plt.figure()
plt.xlabel("k")
plt.ylabel("accuracy")
plt.scatter(k_range, scores)
plt.xticks([0, 5, 10, 15, 20])
plt.show()
