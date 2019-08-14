# KNN Application on Avocado dataset

#Data Preprocessing Parts
#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

#Required Functions
def euclideanDistance(x,y):
    return math.sqrt((x[0]-y[0])*(x[0]-y[0]) + (x[1]-y[1])*(x[1]-y[1]))

def find5MinElements(arr):
    temp = []
    for i in arr:
        temp.append(i)
    temp.sort()
    return temp[:5]



# Importing the dataset
dataset = pd.read_csv('avocado_dataset.csv')
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 11].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

y_train_string = []
for i in y_train:
    y_train_string.append(str(i))

# Use standard normalization to normalize Independent Variables
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Apply KNN to classify test set results

#We will look at 5 nearest neighbors to determine if its organic avocado or not

#Create Pred Matrix with test set
pred_matrix=[]
for i in X_test:
    distance_matrix = []
    for j in X_train:
        distance = euclideanDistance(i,j)
        distance_matrix.append(distance)
    nearests = find5MinElements(distance_matrix)
    IndexesOfNearests = []
    for idx in range(5):
        IndexesOfNearests.append(distance_matrix.index(nearests[idx]))
        
    organicCount = 0
    conventionalCount = 0
    for x in IndexesOfNearests:
        if y_train_string[x] == "conventional":
            conventionalCount = conventionalCount + 1
        else:
            organicCount = organicCount + 1
    if conventionalCount > organicCount:
        pred_matrix.append("conventional")
    else:
        pred_matrix.append("organic")


    
print("end")

y_test_string = []
for i in y_test:
    y_test_string.append(str(i))


corrects = 0
errors = 0

for i in range(800):
    if pred_matrix[i] == y_test_string[i] :
        corrects = corrects +1
    else:
        errors = errors +1

print("Corrects = " + str(corrects))
print("Errors = " + str(errors))

#Print Resutls as %Percent
print("Correctness Percentage = %" + str((corrects * 100)/800))
print("Error Percentage = %"+ str((errors * 100)/800))

