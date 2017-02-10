import sys
import os
path=os.getcwd()
path=path.strip('new_idea')
sys.path.append(path)
import pandas
from helper import svm,knn,lda,sgd,dtree,misc_helper
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')
sets = [['a','m','n','s','t','g','q'],['b','e','c','o','x'],['h','k','u','v'],
['d','r','p'],['f'],['l'],['i'],['w'],['y']]

train_data = pandas.read_csv('datasets/set_train.csv')
train_data = train_data.drop('label',axis=0)
train_data = train_data.drop('set_label',axis=0)
model,scalar = svm.get_model(train_data)
pickle.dump(model,open('models/level1model.sav','wb+'))
pickle.dump(scalar,open('scalars/level1scalar.sav','wb+'))
test = pandas.read_csv('datasets/main_test.csv')
test_data = test
actual = pandas.read_csv('datasets/set_test.csv')['label'].values
test_data = test_data.drop(['label'],axis=1).values
test_data = scalar.transform(test_data)
predictions = model.predict(test_data)
print(accuracy_score(actual,predictions))
data = np.column_stack((test_data,predictions))
level1 = pandas.DataFrame(data,columns = train_data.columns.values)
level1.to_csv('datasets/level1.csv',index=False)
for i in range(len(sets)):
    curr_set = sets[i]
    if len(curr_set) > 1:
        curr_frame = level1[level1['label'] == i]
        curr_frame.to_csv('datasets/set_'+str(i)+'_test.csv',index=False)
        curr_frame = test[test['label'].isin(curr_set)]
        curr_frame.to_csv('datasets/set_'+str(i)+'_actual.csv',index=False)

print("Level 1 dataset created")
