import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from helper import knn,lda,dtree,misc_helper,sgd,svm

initialData = pandas.read_csv('../CSV_Data/dataset_6.csv')
allData = initialData
all_features = allData.columns.values
sets = [['a','m','n','s','t','q','o','g','x'],['b','e','c'],['h','k','u','v'],['d','r','p']]
feature_lists = [all_features,all_features,all_features,all_features]

for curr_set,feature_list in zip(sets,feature_lists):
    svm_sum_acc = 0
    knn_sum_acc = 0
    lda_sum_acc = 0
    sgd_sum_acc = 0
    dtree_sum_acc = 0
    svm_conf_matrix = [[0 for x in range(len(curr_set))] for y in range(len(curr_set))]
    knn_conf_matrix = [[0 for x in range(len(curr_set))] for y in range(len(curr_set))]
    lda_conf_matrix = [[0 for x in range(len(curr_set))] for y in range(len(curr_set))]
    sgd_conf_matrix = [[0 for x in range(len(curr_set))] for y in range(len(curr_set))]
    dtree_conf_matrix = [[0 for x in range(len(curr_set))] for y in range(len(curr_set))]
    for i in range(100):
        svm_acc,svm_conf= svm.get_set_accuracy(allData,curr_set,feature_list)
        knn_acc,knn_conf= knn.get_set_accuracy(allData,curr_set,feature_list)
        dtree_acc,dtree_conf= dtree.get_set_accuracy(allData,curr_set,feature_list)
        lda_acc,lda_conf= lda.get_set_accuracy(allData,curr_set,feature_list)
        sgd_acc,sgd_conf= sgd.get_set_accuracy(allData,curr_set,feature_list)
        svm_sum_acc = svm_sum_acc + svm_acc
        knn_sum_acc = knn_sum_acc + knn_acc
        lda_sum_acc = lda_sum_acc + lda_acc
        dtree_sum_acc = dtree_sum_acc + dtree_acc
        sgd_sum_acc = sgd_sum_acc + sgd_acc
        svm_conf_matrix = np.add(svm_conf_matrix,svm_conf)
        knn_conf_matrix = np.add(knn_conf_matrix,knn_conf)
        dtree_conf_matrix = np.add(dtree_conf_matrix,dtree_conf)
        lda_conf_matrix = np.add(lda_conf_matrix,lda_conf)
        sgd_conf_matrix = np.add(sgd_conf_matrix,sgd_conf)
    svm_sum_acc = svm_sum_acc/100
    knn_sum_acc = knn_sum_acc/100
    lda_sum_acc = lda_sum_acc/100
    sgd_sum_acc = sgd_sum_acc/100
    dtree_sum_acc = dtree_sum_acc/100
    print(curr_set)
    print("SVM : " + str(svm_sum_acc))
    print(svm_conf_matrix)
    print("KNN : " + str(knn_sum_acc))
    print(knn_conf_matrix)
    print("DTREE : " + str(dtree_sum_acc))
    print(dtree_conf_matrix)
    print("SGD : " + str(sgd_sum_acc))
    print(sgd_conf_matrix)
    print("LDA : " + str(lda_sum_acc))
    print(lda_conf_matrix)
