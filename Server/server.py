import subprocess
from bottle import run, post, request, response, get, route
from bottle import static_file
from sklearn.neighbors import KNeighborsClassifier
# import pprint

data = []
labels = []
train_data = []
train_labels = []
test_data = []
test_labels = []
clf= KNeighborsClassifier()

from sklearn.cross_validation import train_test_split

@route('/',method='GET')
def index():
    return static_file('index.html',root='static/')

@route('/predictor',method='GET')
def predictor():
    return static_file('predictor.html',root='static/')

@route('/visualizer',method='GET')
def predictor():
    return static_file('visualizer.html',root='static/')

@route('/recorder',method='GET')
def predictor():
    return static_file('recorder.html',root='static/')

@route('/static/<filename>',method='GET')
def serve_static(filename):
    return static_file(filename,root='static/')
@route('/train',method = 'GET')
def train():
    with open("../Collected Data/Rishav/a_f_rishav.csv", 'r') as ppd:
        for line in ppd:
            attr = line.split(',')
            data.append(list(map(float, attr[0:31])))
            labels.append(attr[31][0:-1])

    with open("../Collected Data/Rishav/g_m_rishav.csv", 'r') as ppd:
        for line in ppd:
            attr = line.split(',')
            data.append(list(map(float, attr[0:31])))
            labels.append(attr[31][0:-1])
    global train_data
    global train_labels
    global test_data
    global test_labels
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size = .5)
    # print(test_data)

    clf.fit(train_data,train_labels)
    return "Trained!"
@route('/predict',method = 'POST')
def predict():
    test_data = request.json['ar'];
    # print(test_data)
    predictions = clf.predict(test_data)
    # print(predictions)
    return predictions
run(host='localhost', port=8080, debug=True)
