import pandas
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
dataFrame = pandas.read_csv('../CSV_Data/dataset_1.csv')
uniqueLabels = dataFrame['label'].unique()
target = dataFrame['label'].values
dataFrame = dataFrame.drop('label',axis=1).values
train,test,train_target,test_target = train_test_split(dataFrame,target,test_size = 0.2,stratify=target)
model = DecisionTreeClassifier()
model.fit(train,train_target)
print("Training Error : "+ str(model.score(train,train_target)))
predictions = model.predict(test)
print("Test Error : " + str(accuracy_score(test_target,predictions)))
print(" Confusion Matrix : ")
print(confusion_matrix(test_target, predictions,labels =uniqueLabels))
