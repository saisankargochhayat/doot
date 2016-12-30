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
    for i in range(50):
        svm_sum_acc= svm_sum_acc + svm.get_set_accuracy(allData,curr_set,feature_list)[0]
        dtree_sum_acc= dtree_sum_acc+dtree.get_set_accuracy(allData,curr_set,feature_list)[0]
        sgd_sum_acc= sgd_sum_acc +sgd.get_set_accuracy(allData,curr_set,feature_list)[0]
        lda_sum_acc= lda_sum_acc +lda.get_set_accuracy(allData,curr_set,feature_list)[0]
        knn_sum_acc= knn_sum_acc +knn.get_set_accuracy(allData,curr_set,feature_list)[0]
    svm_sum_acc = svm_sum_acc/50
    knn_sum_acc = knn_sum_acc/50
    lda_sum_acc = lda_sum_acc/50
    sgd_sum_acc = sgd_sum_acc/50
    dtree_sum_acc = dtree_sum_acc/50
    print(curr_set)
    print("SVM : " + str(svm_sum_acc))
    print("KNN : " + str(knn_sum_acc))
    print("DTREE : " + str(dtree_sum_acc))
    print("SGD : " + str(sgd_sum_acc))
    print("LDA : " + str(lda_sum_acc))
