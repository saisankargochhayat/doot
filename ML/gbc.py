import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing, svm
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier()
dataFrame = pandas.read_csv('../CSV_Data/dataset_7.csv')
#print(dataFrame)
target =np.array(dataFrame['label'])

features = np.array(dataFrame.drop(['label'],1))

sum = 0
for i in range(100):
    train_feature, test_feature, train_target, test_target =train_test_split(features, target, test_size=0.2,stratify=target)
    model.fit(train_feature, train_target)
    predict = model.predict(test_feature)
    acc = accuracy_score(test_target,predict)
    sum = sum+acc

print('dataset 7:')
print(sum/100)
