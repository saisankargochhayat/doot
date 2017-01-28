from tornado import websocket, web, ioloop
import os
path=os.getcwd()
path=path.strip('Research') + 'ML'
import sys
sys.path.append(path)
import json
import pandas
import numpy as np
from helper import svm,knn,dtree,sgd,lda

data = []
labels = []
# svm_model , svm_scaler = svm.get_model(dataFrame)
# knn_model , knn_scaler = knn.get_model(dataFrame)
# sgd_model , sgd_scaler = sgd.get_model(dataFrame)
# dtree_model , dtree_scaler = dtree.get_model(dataFrame)
# lda_model , lda_scaler = lda.get_model(dataFrame)
# qda_model , qda_scaler = qda.get_model(dataFrame)

class HomeHandler(web.RequestHandler):
    def get(self):
        self.render("static/research.html")

class Prediction(websocket.WebSocketHandler):
    def check_origin(self, origin):
        return True

    def open(self):
        print("WebSocket opened")

    def on_message(self, message):
        msg = json.loads(message)
        self.write_message(predictions)

    def on_close(self):
        print("WebSocket closed")

app = web.Application([
    (r'/assets/(.*)', web.StaticFileHandler, {'path': 'static/assets/'}),
    (r"/",HomeHandler),
    (r"/ws",Prediction),
    ])

if __name__ == '__main__':
    app.listen(8080)
    print("Listening at 127.0.0.1:8080")
    ioloop.IOLoop.instance().start()
