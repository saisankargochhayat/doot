import subprocess
from bottle import run, post, request, response, get, route
from bottle import static_file
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from data_loader import data_loader
# import pprint

data = []
labels = []

knn_model= KNeighborsClassifier()
svm_model = svm.SVC()
sgd_model = SGDClassifier(loss="hinge", penalty="l2")
dtree_model = tree.DecisionTreeClassifier()

# data_loader is a function that loads data from the given csv files
data,labels = data_loader()

knn_model.fit(data,labels)
svm_model.fit(data,labels)
sgd_model.fit(data,labels)
dtree_model.fit(data,labels)
print("Trained")

@route('/',method='GET')
def index():
    return static_file('index.html',root='static/')

@route('/predictor',method='GET')
def predictor():
    return static_file('predictor.html',root='static/')

@route('/visualizer',method='GET')
def predictor():
    return static_file('visualizer.html',root='static/')


@route('/static/<filename>',method='GET')
def serve_static(filename):
    return static_file(filename,root='static/')
@route('/train',method = 'GET')
def train():
    return "Trained"

@route('/predict',method = 'POST')
def predict():
    test_data = request.json['ar'];
    # print(test_data)
    predictions = {};
    predictions['knn'] = str(knn_model.predict(test_data)[0])
    predictions['svm'] = str(svm_model.predict(test_data)[0])
    predictions['sgd'] = str(sgd_model.predict(test_data)[0])
    predictions['dtree'] = str(dtree_model.predict(test_data)[0])
    # print(predictions)
    return predictions
run(host='localhost', port=8080, debug=True)
