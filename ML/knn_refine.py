import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from helper import knn
dataFrame = pandas.read_csv('../CSV_Data/dataset_6.csv')
weights = ['uniform','distance']
for w in weights:
    sum_acc = 0
    for i in range(50):
        acc,confusion = knn.find_accuracy(dataFrame,5,w)
        sum_acc = sum_acc+ acc
    acc = sum_acc/50
    print(str(w) + " " +str(acc))
