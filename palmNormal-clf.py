from sklearn import tree

train_data = []
train_labels = []

with open("palmpositiondata.csv", 'r') as ppd:
    for line in ppd:
        attr = line.split(',')
        train_data.append(map(float, attr[0:3]))
        train_labels.append(attr[3][0:-1])

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_labels)

inp = raw_input().split()
print clf.predict([map(float, inp)])
