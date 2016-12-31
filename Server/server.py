from tornado import websocket, web, ioloop
import os
path=os.getcwd()
path=path.strip('Server') + 'ML'
import sys
sys.path.append(path)
import tornado.escape
from tornado import gen
import tornado.httpserver
import tornado.options
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.linear_model import SGDClassifier
from sklearn import svm
# from data_loader import data_loader
import json
import pprint
import pandas
from sklearn import svm
import numpy as np
from  tornado.escape import json_decode
from  tornado.escape import json_encode
from feature_extracter_live import *
from sklearn import preprocessing
from helper import svm,knn,dtree,sgd,lda
# define("port", default=8080, help="run on the given port", type=int)

data = []
labels = []
dataFrame = pandas.read_csv('../CSV_Data/dataset_6.csv')
svm_model , svm_scaler = svm.get_model(dataFrame)
knn_model , knn_scaler = knn.get_model(dataFrame)
sgd_model , sgd_scaler = sgd.get_model(dataFrame)
dtree_model , dtree_scaler = dtree.get_model(dataFrame)
lda_model , lda_scaler = lda.get_model(dataFrame)
print("Trained")

class HomeHandler(web.RequestHandler):
    def get(self):
        self.render("static/index.html")

class Predictor(web.RequestHandler):
    def get(self):
        self.render("static/predictor.html")

class Visualizer(web.RequestHandler):
    def get(self):
        self.render("static/visualizer.html")

class Predict(websocket.WebSocketHandler):
    def check_origin(self, origin):
        return True

    def open(self):
        print("WebSocket opened")

    def on_message(self, message):
        msg = json.loads(message)
        test=extract_array(msg)
        predictions = {};
        predictions['svm'] = str(svm_model.predict(svm_scaler.transform(test))[0])
        predictions['knn'] = str(knn_model.predict(knn_scaler.transform(test))[0])
        predictions['lda'] = str(lda_model.predict(lda_scaler.transform(test))[0])
        predictions['sgd'] = str(sgd_model.predict(sgd_scaler.transform(test))[0])
        predictions['dtree'] = str(dtree_model.predict(dtree_scaler.transform(test))[0])
        self.write_message(predictions)

    def on_close(self):
        print("WebSocket closed")

app = web.Application([
    (r'/static/(.*)', web.StaticFileHandler, {'path': 'static/'}),
    (r"/",HomeHandler),
    (r"/predictor",Predictor),
    (r"/visualizer",Visualizer),
    (r"/ws",Predict),
    ])

if __name__ == '__main__':
    app.listen(8080)
    print("Listening at 127.0.0.1:8080")
    ioloop.IOLoop.instance().start()
