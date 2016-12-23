import pandas
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
dataFrame = pandas.read_csv('../CSV_Data/dataset_6.csv')
uniqueLabels = dataFrame['label'].unique()
target = dataFrame['label'].values
dataFrame = dataFrame.drop('label',axis=1).values
dataFrame = preprocessing.scale(dataFrame)
sum_acc = 0
confusion = [[0 for x in range(24)] for y in range(24)]
for i in range(200):
    train,test,train_target,test_target = train_test_split(dataFrame,target,test_size = 0.2,stratify=target)

    model = DecisionTreeClassifier()
    model.fit(train,train_target)
    # print("Training Accuracy : "+ str(model.score(train,train_target)))
    predictions = model.predict(test)
    sum_acc = sum_acc+ accuracy_score(test_target,predictions)
    confusion = np.add(confusion,confusion_matrix(test_target, predictions,labels =uniqueLabels))
    # print("Test Accuracy : " + str(accuracy_score(test_target,predictions)))
    # print(" Confusion Matrix : ")
    # print(confusion_matrix(test_target, predictions,labels =uniqueLabels))
print(sum_acc/200)
print(confusion)
