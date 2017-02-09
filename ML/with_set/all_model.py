import pandas
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import os
path=os.getcwd()

path=path.strip('with_set')
import sys
sys.path.append(path)
from helper import svm,sgd,lda,ridge,knn,dtree
import glob
names = ['rishi','sandy']
sets = [['m','n'],['a','t','s','g','x','o','g'],['b','e','c'],['h','k','u','v'],['d','r','p']]
path = os.path.split(path)[0]
path =path.strip('ML')
path = path+'/CSV_Data/set_experiment/try.csv'
# allfiles = glob.glob(path+"/*")


for filename in range(1):
    allData = pandas.read_csv(path)
    feature_list = allData.columns.values
    print(filename)
    for curr_set in sets:
        svm_sum_acc = 0
        knn_sum_acc = 0
        lda_sum_acc = 0
        sgd_sum_acc = 0
        dtree_sum_acc = 0
        for j in range(100):
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
        svm_sum_acc = svm_sum_acc/100
        knn_sum_acc = knn_sum_acc/100
        lda_sum_acc = lda_sum_acc/100
        sgd_sum_acc = sgd_sum_acc/100
        dtree_sum_acc = dtree_sum_acc/100
        print(curr_set)
        print("SVM : " + str(svm_sum_acc))
        print("KNN : " + str(knn_sum_acc))
        print("DTREE : " + str(dtree_sum_acc))
        print("SGD : " + str(sgd_sum_acc))
        print("LDA : " + str(lda_sum_acc))
