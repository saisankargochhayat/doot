import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from helper import svm
dataFrame = pandas.read_csv('../CSV_Data/try.csv')
c_list = np.logspace(-10,20,num=31)
sum_acc = 0
for i in range(50):
    acc,confusion = svm.find_accuracy(dataFrame,'linear')
    sum_acc = sum_acc+ acc
acc = sum_acc/50
print(str(acc))
for c in c_list:
    sum_acc = 0
    for i in range(50):
        acc,confusion = svm.find_accuracy(dataFrame,'poly',3,c)
        sum_acc = sum_acc+ acc
    acc = sum_acc/50
    print(str(c) + " " +str(acc))
