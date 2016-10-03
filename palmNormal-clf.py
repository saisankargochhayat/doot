from sklearn import tree
import pprint

data = []
labels = []
train_data = []
train_labels = []
test_data = []
test_labels = []
data1 =[]
labels1 =[]
with open("a_f_train.csv", 'r') as ppd:
    for line in ppd:
        attr = line.split(',')
        data.append(map(float, attr[0:31]))
        labels.append(attr[31][0:-1])

with open("a_f_test.csv", 'r') as ppd:
    for line in ppd:
        attr = line.split(',')
        data1.append(map(float, attr[0:31]))
        labels1.append(attr[31][0:-1])

#from sklearn.cross_validation import train_test_split
#train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size = .5)

clf = tree.DecisionTreeClassifier()
clf.fit(data,labels)
predictions = clf.predict(data1)

#with open("data1", 'w') as _file:
#        _file.write(pprint.pformat(data1))

from sklearn.metrics import accuracy_score
print accuracy_score(labels1, predictions)
