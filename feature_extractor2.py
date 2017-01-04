# Import Modules and prepare database connection
import pandas
from pymongo import MongoClient
client = MongoClient()
db = client.doot
import numpy as np
#-------------------------------Function to manipulate Features------------------
# This function takes dataframe and normalized frame to extract the required
# Features from the JSON data and returns the array
def extract_array(frame,norm_frame):
    hand = frame['hands'][0]
    norm_hand = norm_frame['hands'][0]
    data = []
    data.append(hand['pinchStrength'])
    data.append(hand['grabStrength'])
    # Angle between metacarpal,proximal and proximal,intermediate for every finger
    # Through expreiment, it was found that frame['pointables'] is an array of pointables
    # in the following format : thumb, index, middle, ring, pinky
    for pointable in frame['pointables']:
        data.append(mc_prox_angle(pointable,hand))
        data.append(prox_inter_angle(pointable,hand))
    # Angle between consecutive fingers
    for i in range(4):
        data.append(finger_angle(frame['pointables'][i],frame['pointables'][i+1]))
    # Normalized tip distance from palmCenter
    # PalmPosition is the 3d vector of palmCenter only
    # for pointable,norm_pointable in zip(frame['pointables'],norm_frame['pointables']):
    #     tip_vector = np.subtract(hand['palmPosition'],pointable['tipPosition'])
    #     tip_distance = np.linalg.norm(tip_vector)
    #     norm_tip_vector = np.subtract(norm_hand['palmPosition'],norm_pointable['tipPosition'])
    #     norm_tip_distance = np.linalg.norm(norm_tip_vector)
    #     normalized_distance = tip_distance/norm_tip_distance
    #     # Normalized distance shud be between 0 to 1
    #     # But if it is slightly above 1 due to recording error , it can be assigned to 1
    #     if(normalized_distance > 1):
    #         normalized_distance=1
    #     data.append(normalized_distance)
    direction = 0
    if hand['palmNormal'][1]<-0.7:
        direction=1
    elif hand['palmNormal'][1]>0.7:
        direction=2
    else:
        if hand['palmNormal'][0]<-0.7:
            direction=3
        elif hand['palmNormal'][0]>0.7:
            direction=4
        else:
            if hand['palmNormal'][2]<-0.7:
                direction=5
            elif hand['palmNormal'][2]>0.7:
                direction=6
    data.append(direction)
    #Append the label
    for pointable in frame['pointables']:
        data.append(pointable['direction'][0])
        data.append(pointable['direction'][1])
        data.append(pointable['direction'][2])
    data.append(frame['label'])
    return data

#---------------------------------------------------------------------------------
# Used to calculate Angle between metacarpal bone and proximal bone given
# Pointable and hand objects
def mc_prox_angle(pointable,hand):
    a = np.subtract(hand['wrist'],pointable['mcpPosition'])
    b = np.subtract(pointable['mcpPosition'],pointable['pipPosition'])
    return getAngle(a,b)

# Used to calculate Angle between proximal bone and intermediate bone given
# Pointable and hand objects
def prox_inter_angle(pointable,hand):
    a = np.subtract(pointable['mcpPosition'],pointable['pipPosition'])
    b = np.subtract(pointable['pipPosition'],pointable['dipPosition'])
    return getAngle(a,b)
# Used to calculate Angle 2 fingers
def finger_angle(x,y):
    a = np.subtract(x['mcpPosition'],x['pipPosition'])
    b = np.subtract(y['mcpPosition'],y['pipPosition'])
    # I am still confused with this thing.
    # For calulating angle between fingers , should we use the vector between
    # knuckle of finger and tip or
    # knuckle of finger and mcpPosition
    # The above commented Lines prove that they are pretty Different In some cases
    # p = np.subtract(x['mcpPosition'],x['tipPosition'])
    # q = np.subtract(y['mcpPosition'],y['tipPosition'])
    # print(getAngle(a,b)-getAngle(p,q))
    return getAngle(a,b)

# A function to get angle between any 2 vectors
def getAngle(x,y):
    x = np.array(x)
    y = np.array(y)
    cosang = np.dot(x,y)
    sinang = np.linalg.norm(np.cross(x,y))
    angle = np.arctan2(sinang,cosang)
    return np.degrees(angle)

# This frame checks for validity of the frame. It can have more conditions than current
def validate_frame(frame):
    if frame:
        if 'hands' in frame:
            if len(frame['hands']) > 0:
                return True
            else:
                return False
        else:
            return False
    else:
        return False

# Data will contain all the data in array format
data = []
for collection_name in db.collection_names(): # For all the collections in the db
    if(collection_name != 'norm'):              #Except for norm collection
        collection = db[collection_name]
        #Retrieve normalized frame for the current user
        norm_frame = db['norm'].find_one({'username':collection_name})
        if validate_frame(norm_frame):
            for frame in collection.find():
                if validate_frame(frame):
                    curr_array = extract_array(frame,norm_frame)
                    data.append(curr_array)
# Data now contains all the dateset in array format
print("Length of Dataset : " + str(len(data)))
# column_names contains the name of every column
# It is used to create the pandas dataframe
column_names = ['pinchStrength','grabStrength']
finger_map = ['thumb','index','middle','ring','pinky']
for i in range(5):
    column_names.append(finger_map[i]+'_meta_proxi')
    column_names.append(finger_map[i]+'_proxi_inter')
for i in range(4):
    column_names.append(finger_map[i]+'_'+finger_map[i+1])
# for i in range(5):
#     column_names.append(finger_map[i]+'_'+'center_distance')
column_names.append('palm_direction')
for i in range(5):
    column_names.append(finger_map[i]+"_direction_x")
    column_names.append(finger_map[i]+"_direction_y")
    column_names.append(finger_map[i]+"_direction_z")
column_names.append('label')
# Convert to pandas Dataframe
data_df = pandas.DataFrame(data,columns=column_names)
# Write to csv File
data_df.to_csv('CSV_Data/dataset_0.csv',index=False)
print("Successfully Created CSV file")
