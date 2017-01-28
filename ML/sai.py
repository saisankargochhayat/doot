import pandas
from random import randint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from helper import svm,sgd,knn,dtree,lda
import numpy as np
setlist = [['a','m','n','s','t','g','q','x','o'],['b','e','c'],['h','k','u','v'],
['d','r','p'],['f'],['l'],['i'],['w'],['y']]

initialData = pandas.read_csv('../CSV_Data/dataset_6.csv')
allData = initialData
matrix = allData.drop('label',axis=1).values
matrix = preprocessing.scale(matrix)
allData = pandas.DataFrame(matrix,columns = initialData.columns.drop('label'))
allData['label'] = initialData['label']
sum_svm_acc=0
sum_knn_acc=0
sum_dtree_acc=0
sum_sgd_acc=0
sum_lda_acc=0
sum_svm_confusion = [[0 for x in range(9)] for y in range(9)]
sum_knn_confusion = [[0 for x in range(9)] for y in range(9)]
sum_dtree_confusion = [[0 for x in range(9)] for y in range(9)]
sum_lda_confusion = [[0 for x in range(9)] for y in range(9)]
sum_sgd_confusion = [[0 for x in range(9)] for y in range(9)]
for i in range(100):
    master_set = []
    for curr_set in setlist:
        master_set.append(curr_set[randint(0,len(curr_set)-1)])
    currentData = allData[allData['label'].isin(master_set)]

    acc,con = svm.find_accuracy(currentData)
    sum_svm_acc = sum_svm_acc + acc
    sum_svm_confusion = np.add(sum_svm_confusion,con)

    acc,con = sgd.find_accuracy(currentData)
    sum_sgd_acc = sum_sgd_acc + acc
    sum_sgd_confusion = np.add(sum_sgd_confusion,con)

    acc,con = knn.find_accuracy(currentData)
    sum_knn_acc = sum_knn_acc + acc
    sum_knn_confusion = np.add(sum_knn_confusion,con)

    acc,con = lda.find_accuracy(currentData)
    sum_lda_acc = sum_lda_acc + acc
    sum_lda_confusion = np.add(sum_lda_confusion,con)

    acc,con = dtree.find_accuracy(currentData)
    sum_dtree_acc = sum_dtree_acc + acc
    sum_dtree_confusion = np.add(sum_dtree_confusion,con)

print("SVM : ")
print(sum_svm_confusion)
print("KNN : ")
print(sum_knn_confusion)
print("Dtree : ")
print(sum_dtree_confusion)
print("LDA : ")
print(sum_lda_confusion)
print("SGD : ")
print(sum_sgd_confusion)


print("SVM : ")
print(sum_svm_acc/100)
print("KNN : ")
print(sum_knn_acc/100)
print("Dtree : ")
print(sum_dtree_acc/100)
print("LDA : ")
print(sum_lda_acc/100)
print("SGD : ")
print(sum_sgd_acc/100)
