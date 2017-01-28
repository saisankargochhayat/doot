import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from helper import dtree
dataFrame = pandas.read_csv('../CSV_Data/dataset_6.csv')

c_list = np.logspace(-10,20,num=31)
for c in c_list:
    sum_acc = 0
    for i in range(50):
        acc,confusion = dtree.find_accuracy(dataFrame,c)
        sum_acc = sum_acc+ acc
    acc = sum_acc/50
    print(str(c) + " " + str(acc))
