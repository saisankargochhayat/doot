from tornado import websocket, web, ioloop
import os
path=os.getcwd()
path=path.strip('Research') + 'ML'
import sys
sys.path.append(path)
import json
import pandas
import numpy as np
from helper import svm,knn,dtree,sgd,lda,gbc,ridge,lr,qda

model_dict = {
"SVM":svm,
"KNN":knn,
"DTREE":dtree,
"SGD":sgd,
"LDA":lda,
"GBC":gbc,
"RIDGE":ridge,
"LR":lr,
"QDA":qda,
}

class HomeHandler(web.RequestHandler):
    def get(self):
        self.render("static/research.html")

class Prediction(websocket.WebSocketHandler):
    def check_origin(self, origin):
        return True

    def open(self):
        print("WebSocket opened")

    def on_message(self, message):
        msg=json.loads(message)
        dataset = msg['dataset']
        model_name = msg['model']
        letters = msg['alphabets']
        features = msg['features']
        model = model_dict[model_name]
        dataFrame = pandas.read_csv("../CSV_Data/"+dataset)
        if feature_check(dataFrame, features):
            sum_acc = 0
            for i in range(100):
                acc,confusion = model.get_set_accuracy(dataFrame,letters,features)
                sum_acc = sum_acc + acc
            acc = { "accuracy" : sum_acc/100}
            self.write_message(acc)
        else:
            acc = { "accuracy" : "Some features were not found in dataset"}
            self.write_message(acc)

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
