from tornado import websocket, web, ioloop
import json
from pymongo import MongoClient
client = MongoClient()
# Change to record in database for live server or for research database
db = client.doot_3d
# db = client.doot
class IndexHandler(web.RequestHandler):
    def get(self):
        self.render("static/recorder.html")

class SocketHandler(websocket.WebSocketHandler):
    def check_origin(self, origin):
        return True

    def open(self):
        print("WebSocket opened")

    def on_message(self, message):
        msg = json.loads(message)
        data = msg['frame']
        user = msg['user']
        collection = db[user]
        collection.insert_one(json.loads(data))

    def on_close(self):
        print("WebSocket closed")

app = web.Application([
    (r'/', IndexHandler),
    (r'/static/(.*)', web.StaticFileHandler, {'path': 'static/'}),
    (r'/ws', SocketHandler)
    ])

if __name__ == '__main__':
    app.listen(8080)
    print("Listening at 127.0.0.1:8080")
    ioloop.IOLoop.instance().start()
