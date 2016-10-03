from sklearn import tree

data = []
labels = []
train_data = []
train_labels = []
test_data = []
test_labels = []

with open("palmpositiondata.csv", 'r') as ppd:
    for line in ppd:
        attr = line.split(',')
        data.append(map(float, attr[0:3]))
        labels.append(attr[3][0:-1])

from sklearn.cross_validation import train_test_split
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size = .5)

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_labels)
predictions = clf.predict(test_data)

from sklearn.metrics import accuracy_score
print accuracy_score(test_labels, predictions)
