from sklearn import tree

train_data = []
train_labels = []
test_data = []
test_labels = []

with open("palmpositiondata.csv", 'r') as ppd:
    for line in ppd:
        attr = line.split(',')
        train_data.append(map(float, attr[0:3]))
        train_labels.append(attr[3][0:-1])

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_labels)

with open("palmpositiontest.csv", 'r') as ppt:
    for line in ppt:
        attr = line.split(',')
        test_data.append(map(float, attr[0:3]))
        test_labels.append(attr[3][0:-1])

predictions = clf.predict(test_data)
for i in range(len(predictions)):
    print predictions[i], test_labels[i] 

from sklearn.metrics import accuracy_score
print accuracy_score(test_labels, predictions)
