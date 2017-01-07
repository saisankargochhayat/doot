import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from helper import knn,svm,dtree,sgd,lda,misc_helper
import collections
import string

dataFrame = pandas.read_csv('../CSV_Data/dataset_6.csv')
# dataFrame = dataFrame[~dataFrame['label'].isin(['m','n'])]
sum_acc = 0
print(svm.get_precision(dataFrame))
# for i in range(100):
#     train,test = train_test_split(dataFrame,stratify=dataFrame['label'])
#     test,test_target = misc_helper.split_feature_target(test)
#     svm_model,svm_scaler = svm.get_model(train)
#     knn_model,knn_scaler = knn.get_model(train)
#     dtree_model,dtree_scaler = dtree.get_model(train)
#     lda_model,lda_scaler = lda.get_model(train)
#     sgd_model,sgd_scaler = sgd.get_model(train)
#     predictions = []
#     confidences = []
#     i=0
#     for curr_test,curr_actual in zip(test,test_target):
#         curr_prediction = {}
#         curr_test = curr_test.reshape(1,-1)
#         svm_test = curr_test
#         svm_test = svm_scaler.transform(svm_test)
#         svm_prediction = svm_model.predict(svm_test)
#
#
#         knn_test = curr_test
#         knn_test = knn_scaler.transform(knn_test)
#         knn_prediction = knn_model.predict(knn_test)
#
#
#         dtree_test = curr_test
#         dtree_test = dtree_scaler.transform(dtree_test)
#         dtree_prediction = dtree_model.predict(dtree_test)
#
#
#         sgd_test = curr_test
#         sgd_test = sgd_scaler.transform(sgd_test)
#         sgd_prediction = sgd_model.predict(sgd_test)
#
#
#         lda_test = curr_test
#         lda_test = lda_scaler.transform(lda_test)
#         lda_prediction = lda_model.predict(lda_test)
#
#         index = confidences[i].argmax()
#         class_list = list(string.ascii_lowercase)
#         class_list.remove('j')
#         class_list.remove('z')
#         predictions.append(class_list[index])
#         i = i+1
#
#
#     sum_acc = sum_acc + accuracy_score(predictions,test_target)
# print(sum_acc/100)
