from sklearn import tree
import numpy as np

data = []
labels = []
train_data = []
train_labels = []
test_data = []
test_labels = []
test=[0,10,15]
names=['x','y','z']



with open("palmpositiondata.csv", 'r') as ppd:
    for line in ppd:
        attr = line.split(',')
        data.append(map(float, attr[0:3]))
        labels.append(attr[3][0:-1])


data=np.asarray(data)
labels=np.asarray(labels)


#training data
train_data=np.delete(data,test,axis=0)
train_labels=np.delete(labels,test)


#testing data
test_data=data[test]
test_labels=labels[test]



clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_labels)
predictions = clf.predict(test_data)



from sklearn.externals.six import StringIO
import pydot
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,feature_names=names,class_names=labels,filled=True,rounded=True,impurity=False)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("fingers.pdf")
