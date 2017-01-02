import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from helper import svm
dataFrame = pandas.read_csv('../CSV_Data/dataset_6.csv')


sum_acc = 0
sum_confusion = [[0 for x in range(24)] for y in range(24)]
for i in range(10):

    acc,confusion = svm.find_accuracy(dataFrame)
    sum_acc = sum_acc+ acc
    sum_confusion = np.add(sum_confusion,confusion)


print(sum_confusion)
print(sum_acc/10)
