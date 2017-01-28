import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from helper import knn,svm,dtree,sgd,lda,ridge,misc_helper
import collections
import string

dataFrame = pandas.read_csv('../CSV_Data/dataset_8.csv')
svm_precision = svm.get_precision(dataFrame)
knn_precision = knn.get_precision(dataFrame)
sgd_precision = sgd.get_precision(dataFrame)
lda_precision = lda.get_precision(dataFrame)
dtree_precision = dtree.get_precision(dataFrame)
class_list = list(string.ascii_lowercase)
class_list.remove('j')
class_list.remove('z')
print("Found precision")
sum_acc = 0
for i in range(100):
    train,test = train_test_split(dataFrame,stratify=dataFrame['label'],test_size=0.2)
    test,test_target = misc_helper.split_feature_target(test)
    svm_model,scalar = svm.get_model(train)
    knn_model,knn_scaler = knn.get_model(train)
    dtree_model,dtree_scaler = dtree.get_model(train)
    lda_model,lda_scaler = lda.get_model(train)
    sgd_model,sgd_scaler = sgd.get_model(train)
    predictions = []
    for curr_test in test:
        curr_prediction = [0.0 for x in range(24)]
        curr_test = np.array(curr_test)
        curr_test = curr_test.reshape(1,-1)
        curr_test = scalar.transform(curr_test)

        svm_prediction = svm_model.predict(curr_test)
        curr_prediction = np.add(curr_prediction,svm_precision[class_list.index(svm_prediction[0])])

        knn_prediction = knn_model.predict(curr_test)
        curr_prediction = np.add(curr_prediction,knn_precision[class_list.index(knn_prediction[0])])

        sgd_prediction = sgd_model.predict(curr_test)
        curr_prediction = np.add(curr_prediction,sgd_precision[class_list.index(sgd_prediction[0])])

        lda_prediction = lda_model.predict(curr_test)
        curr_prediction = np.add(curr_prediction,lda_precision[class_list.index(lda_prediction[0])])

        dtree_prediction = dtree_model.predict(curr_test)
        curr_prediction = np.add(curr_prediction,dtree_precision[class_list.index(dtree_prediction[0])])

        predictions.append(class_list[np.argmax(curr_prediction)])
    acc = accuracy_score(predictions,test_target)
    sum_acc = sum_acc + acc
print(sum_acc/100)
