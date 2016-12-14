import pandas
from pymongo import MongoClient
client = MongoClient()
db = client.doot

#-------------------------------Function to manipulate Features------------------
def extract_array(frame):
    hand = frame['hands'][0]
    data = []
    data.append(hand['pinchStrength'])
    data.append(hand['grabStrength'])
    print(hand['finger'])
    return data

#---------------------------------------------------------------------------------
def validate_frame(frame):
    if 'hands' in frame:
        if len(frame['hands']) > 0:
            return True
        else:
            return False
    else:
        return False
for collection_name in db.collection_names():
    if(collection_name != 'norm'):
        collection = db[collection_name]
        for frame in collection.find():
            if validate_frame(frame):
                curr_array = extract_array(frame)
