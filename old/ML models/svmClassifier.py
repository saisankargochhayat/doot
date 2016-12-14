from sklearn import tree
from sklearn import svm
#from sklearn.neighbors import KNeighborsClassifier
# import pprint

data = []
labels = []
train_data = []
train_labels = []
test_data = []
test_labels = []

with open("../Server/sampledata.csv", 'r') as ppd:
    for line in ppd:
        attr = line.split(',')
        data.append(attr)

with open("../Server/samplelabel.csv", 'r') as ppd:
    for line in ppd:
        attr = line.split(',')
        labels.append(attr[0][0:len(attr[0])-1])

# with open("a_f_test1.csv", 'r') as ppd:
#     for line in ppd:
#         attr = line.split(',')
#         test_data.append(map(float, attr[0:31]))
#         test_labels.append(attr[31][0:-1])

# with open("a_f_test2.csv", 'r') as ppd:
#     for line in ppd:
#         attr = line.split(',')
#         test_data.append(map(float, attr[0:31]))
#         test_labels.append(attr[31][0:-1])

# with open("a_f_test.csv", 'r') as ppd:
#     for line in ppd:
#         attr = line.split(',')
#         test_data.append(map(float, attr[0:31]))
#         test_labels.append(attr[31][0:-1])


from sklearn.cross_validation import train_test_split
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size = .5)

clf = svm.SVC()
clf.fit(train_data,train_labels)
predictions = clf.predict(test_data)

# for  i in range(len(predictions)):
#     print predictions[i], test_labels[i]

#with open("data1", 'w') as _file:
#        _file.write(pprint.pformat(data1))

from sklearn.metrics import accuracy_score
print(accuracy_score(test_labels, predictions))
