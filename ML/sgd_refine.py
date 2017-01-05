import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from helper import sgd
dataFrame = pandas.read_csv('../CSV_Data/dataset_6.csv')
loss = ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron']
penalty = ['none', 'l2', 'l1', 'elasticnet']
for p in penalty:
    sum_acc = 0
    for i in range(50):
        acc,confusion = sgd.find_accuracy(dataFrame,"hinge",p)
        sum_acc = sum_acc+ acc
    acc = sum_acc/50
    print(str(p) + " " +str(acc))
