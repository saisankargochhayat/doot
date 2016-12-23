import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from helper import knn,svm,dtree,sgd,lda

dataFrame = pandas.read_csv('../CSV_Data/dataset_6.csv')

sum_svm_acc=0
sum_knn_acc=0
sum_dtree_acc=0
sum_sgd_acc=0
sum_lda_acc=0
sum_svm_confusion = [[0 for x in range(24)] for y in range(24)]
sum_knn_confusion = [[0 for x in range(24)] for y in range(24)]
sum_dtree_confusion = [[0 for x in range(24)] for y in range(24)]
sum_lda_confusion = [[0 for x in range(24)] for y in range(24)]
sum_sgd_confusion = [[0 for x in range(24)] for y in range(24)]
for i in range(200):

    svm_acc,svm_confusion = svm.find_accuracy(dataFrame)
    sum_svm_acc = sum_svm_acc+svm_acc
    sum_svm_confusion = np.add(sum_svm_confusion,svm_confusion)

    knn_acc,knn_confusion = knn.find_accuracy(dataFrame)
    sum_knn_acc = sum_knn_acc+knn_acc
    sum_knn_confusion = np.add(sum_knn_confusion,knn_confusion)

    dtree_acc,dtree_confusion = dtree.find_accuracy(dataFrame)
    sum_dtree_acc = sum_dtree_acc+dtree_acc
    sum_dtree_confusion = np.add(sum_dtree_confusion,dtree_confusion)

    sgd_acc,sgd_confusion = sgd.find_accuracy(dataFrame)
    sum_sgd_acc = sum_sgd_acc+sgd_acc
    sum_sgd_confusion = np.add(sum_sgd_confusion,sgd_confusion)

    lda_acc,lda_confusion = lda.find_accuracy(dataFrame)
    sum_lda_acc = sum_lda_acc+lda_acc
    sum_lda_confusion = np.add(sum_lda_confusion,lda_confusion)

sum_svm_acc=sum_svm_acc/200
sum_lda_acc=sum_lda_acc/200
sum_sgd_acc=sum_sgd_acc/200
sum_knn_acc=sum_knn_acc/200
sum_dtree_acc=sum_dtree_acc/200

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
print(sum_svm_acc)
print("KNN : ")
print(sum_knn_acc)
print("Dtree : ")
print(sum_dtree_acc)
print("LDA : ")
print(sum_lda_acc)
print("SGD : ")
print(sum_sgd_acc)
