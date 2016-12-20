import pandas
from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

master_data = pandas.read_csv('../../CSV_Data/dataset_3.csv')
all_data = master_data
setlist = [['a','m','n','s','t','o','g','q'],['b','e','c'],['h','k','u','v']
,['d','r','p'],['f'],['l'],['i'],['w'],['y'],['x']]
set_name_list = ['set_'+str(x) for x in range(len(setlist))]
print(set_name_list)
for curr_set,i in zip(setlist,set_name_list):
    all_data['label'][all_data['label'].isin(curr_set)] = i
target = all_data['label'].values
data_frame = all_data.drop('label',axis=1).values

sum_acc=0;
sum_conf_matrix=sum_svm_confusion = [[0 for x in range(len(set_name_list))] for y in range(len(set_name_list))]
for i in range(50):
    train,test,train_target,test_target = train_test_split(data_frame,target,test_size = 0.2,stratify=target)

    model = svm.SVC(kernel='linear')
    model.fit(train,train_target)
    predictions = model.predict(test)
    sum_acc = sum_acc + accuracy_score(test_target,predictions)
    # print(confusion_matrix(test_target,predictions,labels = set_name_list))
    sum_conf_matrix = np.add(sum_conf_matrix,confusion_matrix(test_target,predictions,labels = set_name_list))
print(sum_acc/50)
print(sum_conf_matrix)
