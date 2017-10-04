import pandas
df = pandas.read_csv('datasets/train.csv')
all_features = df.columns.values

set1 = ['thumb_meta_proxi', 'thumb_proxi_inter',
'index_meta_proxi', 'index_proxi_inter', 'middle_meta_proxi',
'middle_proxi_inter', 'ring_meta_proxi', 'ring_proxi_inter',
'pinky_meta_proxi', 'pinky_proxi_inter', 'thumb_index', 'index_middle',
'middle_ring', 'ring_pinky', 'thumb_center_distance', 'index_center_distance',
'middle_center_distance', 'ring_center_distance', 'pinky_center_distance', 'palm_direction',
'thumb_hand_angle', 'thumb_direction_x', 'thumb_direction_y', 'thumb_direction_z',
'index_hand_angle', 'index_direction_x', 'index_direction_y', 'index_direction_z',
'middle_hand_angle', 'middle_direction_x', 'middle_direction_y', 'middle_direction_z',
'ring_hand_angle', 'ring_direction_x', 'ring_direction_y', 'ring_direction_z',
'pinky_hand_angle', 'pinky_direction_x', 'pinky_direction_y', 'pinky_direction_z']

set2 = ['hand_direction_x', 'hand_direction_y', 'hand_direction_z',
'thumb_direction_x', 'thumb_direction_y', 'thumb_direction_z',
'index_direction_x', 'index_direction_y', 'index_direction_z',
'middle_direction_x', 'middle_direction_y', 'middle_direction_z',
'ring_direction_x', 'ring_direction_y', 'ring_direction_z',
'pinky_direction_x', 'pinky_direction_y', 'pinky_direction_z']
set3 = ['hand_direction_x', 'hand_direction_y', 'hand_direction_z',
'thumb_direction_x', 'thumb_direction_y', 'thumb_direction_z',
'index_direction_x', 'index_direction_y', 'index_direction_z',
'middle_direction_x', 'middle_direction_y', 'middle_direction_z',
'ring_direction_x', 'ring_direction_y', 'ring_direction_z',
'pinky_direction_x', 'pinky_direction_y', 'pinky_direction_z']
set4 = ['hand_direction_x', 'hand_direction_y', 'hand_direction_z',
'thumb_direction_x', 'thumb_direction_y', 'thumb_direction_z',
'index_direction_x', 'index_direction_y', 'index_direction_z',
'middle_direction_x', 'middle_direction_y', 'middle_direction_z',
'ring_direction_x', 'ring_direction_y', 'ring_direction_z',
'pinky_direction_x', 'pinky_direction_y', 'pinky_direction_z']

set_features = [set1,set2,set3,set4]

set_divide_features = ['thumb_meta_proxi', 'thumb_proxi_inter',
'index_meta_proxi', 'index_proxi_inter', 'middle_meta_proxi',
'middle_proxi_inter', 'ring_meta_proxi', 'ring_proxi_inter',
'pinky_meta_proxi', 'pinky_proxi_inter',
'thumb_hand_angle', 'thumb_direction_x', 'thumb_direction_y', 'thumb_direction_z',
'index_hand_angle', 'index_direction_x', 'index_direction_y', 'index_direction_z',
'middle_hand_angle', 'middle_direction_x', 'middle_direction_y', 'middle_direction_z',
'ring_hand_angle', 'ring_direction_x', 'ring_direction_y', 'ring_direction_z',
'pinky_hand_angle', 'pinky_direction_x', 'pinky_direction_y', 'pinky_direction_z']
