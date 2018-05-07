import numpy as np
import pandas as pd
import tensorflow as tf
from model import Model
from genFrame import genFrames

# f = open("location.txt").readlines()
# station_dic = {}
# for line in f:
#     temp = line.split(" ")
#     station_dic[temp[0]] = (float(temp[1]), float(temp[2]))

# frame = pd.read_csv("beijing_17_18_aq.csv")
# station_id = frame["stationId"]
# utc_time = frame["utc_time"]

df_aq_station,df_me_station = genFrames(1, 10)

record_len = np.shape(df_aq_station[0]["day"])[0]

keys = df_aq_station[0].keys()
dtype = []
for key in keys:
    if(key == "day" or key == "hour"):
        dtype.append((key, "int")) 
    else:
        dtype.append((key, "float")) 
print dtype

aq_num = len(df_aq_station)
print record_len

mat = np.zeros([record_len * aq_num, 10])
for i in range(aq_num):
    temp = np.zeros([record_len, 10])
    for j in range(len(df_aq_station[0].keys())):
        temp[:, j] = df_aq_station[i][df_aq_station[0].keys()[j]]
    mat[i * record_len : (i + 1) * record_len] = temp

l = mat[np.argsort(mat[:, 0] * 1e3 + mat[:, 1] + mat[:, 2] * 1e8 + mat[:, 3] * 1e6)]
temp = np.array(l[:10], dtype=float)
print temp

with tf.Session() as sess:
    print "HAHA"

    
