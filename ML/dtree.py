import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from helper import dtree,misc_helper
dataFrame = pandas.read_csv('../CSV_Data/dataset_6.csv')


sum_acc = 0
sum_confusion = [[0 for x in range(24)] for y in range(24)]
for i in range(100):

    acc,confusion = dtree.find_accuracy(dataFrame)
    sum_acc = sum_acc+ acc
    sum_confusion = np.add(sum_confusion,confusion)


misc_helper.write_matrix(sum_confusion,"conf_matrices/dtree_conf.csv")
print(sum_acc/100)
