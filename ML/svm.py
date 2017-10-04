import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import preprocessing
from helper import svm,misc_helper
dataFrame = pandas.read_csv('../CSV_Data/master_dataset.csv')


sum_acc = 0
sum_confusion = np.array([[0 for x in range(24)] for y in range(24)])
for i in range(100):

    acc,confusion = svm.find_accuracy(dataFrame)
    sum_acc = sum_acc+ acc
    sum_confusion = np.add(sum_confusion,confusion)
print(sum_confusion)
sum_confusion = sum_confusion.transpose()
credibility = np.array([0.0 for x in range(24)])
for i in range(len(credibility)):
    # print(sum_confusion[i][i])
    # print("by")
    # print(np.sum(sum_confusion[i]))
    credibility[i] = float(sum_confusion[i][i])/float(np.sum(sum_confusion[i]))
# misc_helper.write_matrix(sum_confusion,"conf_matrices/svm_poly_conf.csv")
# print(sum_confusion)
print(credibility)
print(sum_acc/100)
