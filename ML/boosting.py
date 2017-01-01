import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from helper import knn,svm,dtree,sgd,lda,misc_helper
import collections
import string

dataFrame = pandas.read_csv('../CSV_Data/dataset_6.csv')
dataFrame = dataFrame[~dataFrame['label'].isin(['m','n'])]
sum_acc = 0
for i in range(1):
    train,test = train_test_split(dataFrame,stratify=dataFrame['label'])
    test,test_target = misc_helper.split_feature_target(test)
    svm_model,svm_scaler = svm.get_model(train)
    knn_model,knn_scaler = knn.get_model(train)
    dtree_model,dtree_scaler = dtree.get_model(train)
    lda_model,lda_scaler = lda.get_model(train)
    sgd_model,sgd_scaler = sgd.get_model(train)
    predictions = []
    confidences = []
    i=0
    for curr_test,curr_actual in zip(test,test_target):
        curr_prediction = {}
        curr_test = curr_test.reshape(1,-1)
        svm_test = curr_test
        svm_test = svm_scaler.transform(svm_test)
        svm_prediction = svm_model.predict(svm_test)[0]
        confidences.append(svm_model.predict_proba(svm_test)[0])
        if(svm_prediction in curr_prediction):
            curr_prediction[svm_prediction] = curr_prediction[svm_prediction] + 1
        else :
            curr_prediction[svm_prediction] = 1

        knn_test = curr_test
        knn_test = knn_scaler.transform(knn_test)
        knn_prediction = knn_model.predict(knn_test)[0]
        confidences[i] = confidences[i] + knn_model.predict_proba(knn_test)[0]
        if(knn_prediction in curr_prediction):
            curr_prediction[knn_prediction] = curr_prediction[knn_prediction] + 1
        else :
            curr_prediction[knn_prediction] = 1

        dtree_test = curr_test
        dtree_test = dtree_scaler.transform(dtree_test)
        dtree_prediction = dtree_model.predict(dtree_test)[0]
        confidences[i] = confidences[i] + dtree_model.predict_proba(dtree_test)[0]
        if(dtree_prediction in curr_prediction):
            curr_prediction[dtree_prediction] = curr_prediction[dtree_prediction] + 1
        else :
            curr_prediction[dtree_prediction] = 1

        sgd_test = curr_test
        sgd_test = sgd_scaler.transform(sgd_test)
        sgd_prediction = sgd_model.predict(sgd_test)[0]
        # confidences[i] = confidences[i] + sgd_model.predict_proba(sgd_test)[0]
        if(sgd_prediction in curr_prediction):
            curr_prediction[sgd_prediction] = curr_prediction[sgd_prediction] + 1
        else :
            curr_prediction[sgd_prediction] = 1

        lda_test = curr_test
        lda_test = lda_scaler.transform(lda_test)
        lda_prediction = lda_model.predict(lda_test)[0]
        confidences[i] = confidences[i] + lda_model.predict_proba(lda_test)[0]
        if(lda_prediction in curr_prediction):
            curr_prediction[lda_prediction] = curr_prediction[lda_prediction] + 1
        else :
            curr_prediction[lda_prediction] = 1

        counter = collections.Counter(curr_prediction)
        index = confidences[i].argmax()
        class_list = list(string.ascii_lowercase)
        class_list.remove('j')
        class_list.remove('z')
        class_list.remove('m')
        class_list.remove('n')
        predictions.append(class_list[index])
        i = i+1


    sum_acc = sum_acc + accuracy_score(predictions,test_target)
print(sum_acc/1)
