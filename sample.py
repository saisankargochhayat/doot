from sklearn import tree
features = [[140,1],[130,1],[150,0],[170,0]] #here 140,150,130 ... etc are one feature that is weight, 1 ,1 ,0, 0 are another feature for bumpy and smooth for a apple or an orange.

labels=[0,0,1,1] # here labels 0 n 1 are for  apple and orange...see the number of elements of the features and labels are same....
clf=tree.DecisionTreeClassifier()
clf=clf.fir(features,labels)
print clf.predict([[150,0]]) #this will predict the right ans!
