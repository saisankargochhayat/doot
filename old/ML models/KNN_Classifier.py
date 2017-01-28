# from sklearn import svm
# from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
# import pprint

data = []
labels = []
train_data = []
train_labels = []
test_data = []
test_labels = []

# with open("../Collected Data/a_f_rishav.csv", 'r') as ppd:
#     for line in ppd:
#         attr = line.split(',')
#         data.append(list(map(float, attr[0:31])))
#         labels.append(attr[31][0:-1])
#
# with open("../Collected Data/g_m_rishav.csv", 'r') as ppd:
#     for line in ppd:
#         attr = line.split(',')
#         data.append(list(map(float, attr[0:31])))
#         labels.append(attr[31][0:-1])

with open("../Server/all_data/rishav_all_a_f_data.csv", 'r') as ppd:
    for line in ppd:
        attr = line.split(',')
        data.append(attr)

with open("../Server/all_data/rishav_all_a_f_label.csv", 'r') as ppd:
    for line in ppd:
        attr = line.split(',')
        labels.append(attr[0][0:len(attr[0])-1])
with open("../Server/all_data/lalu_all_a_f_data.csv", 'r') as ppd:
    for line in ppd:
        attr = line.split(',')
        data.append(attr)

with open("../Server/all_data/lalu_all_a_f_label.csv", 'r') as ppd:
    for line in ppd:
        attr = line.split(',')
        labels.append(attr[0][0:len(attr[0])-1])
#
from sklearn.cross_validation import train_test_split
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size = .5)
# print(train_labels)
clf = KNeighborsClassifier()
clf.fit(train_data,train_labels)
predictions = clf.predict(test_data)
#
# for  i in range(len(predictions)):
#     if predictions[i] != test_labels[i]:
#         print predictions[i], test_labels[i]

# with open("data1", 'w') as _file:
#        _file.write(pprint.pformat(data1))

from sklearn.metrics import accuracy_score
print(accuracy_score(test_labels, predictions))
