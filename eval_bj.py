import numpy as np
import pandas as pd
import os
import random
import tensorflow as tf
from model_bj import Model

aq_station_name = []
meo_station_name = []
df_aq_station = []
df_meo_station = []
for file in os.listdir("processedData/beijing"):
    #print file[-6:-1]
    frame = pd.read_csv("processedData/beijing/" + file)
    if(file[-6:-1] == "aq.cs"):
        df_aq_station.append(frame)
        aq_station_name.append(file[:-7])
    else:
        df_meo_station.append(frame)
        meo_station_name.append(file[:-8])

record_len_aq = np.shape(df_aq_station[0]["day"])[0]
record_len_meo = np.shape(df_meo_station[0]["day"])[0]

keys = df_aq_station[0].keys()
dtype = []
for key in keys:
    if(key == "day" or key == "hour" or key == "year"):
        dtype.append((key, "int")) 
    else:
        dtype.append((key, "float")) 

print(meo_station_name)
print(dtype, len(keys))
print(record_len_aq, record_len_meo)
aq_num = len(df_aq_station)
meo_num = len(df_meo_station)

aq_data = []
meo_data = []
for i in range(aq_num):
    temp = np.zeros([record_len_aq, 13])
    for j in range(len(df_aq_station[0].keys())):
        temp[:, j] = df_aq_station[i][df_aq_station[0].keys()[j]]
    aq_data.append(temp)

# for i in range(meo_num):
#     print i
#     temp = np.zeros([record_len_meo, 14])
#     for j in range(len(df_meo_station[0].keys())):
#         temp[:, j] = df_meo_station[i][df_meo_station[0].keys()[j]]
#     meo_data.append(temp)


train_aq_data = []
test_aq_data = []
mean_aq = []
var_aq = []
for data in aq_data:
    temp = data[:-528, -6:]
    train_aq_data.append(temp)
    temp = data[-528:, -6:]
    test_aq_data.append(temp)

train_meo_data = []
test_meo_data = []
mean_meo = np.zeros([6])
var_meo = np.zeros([6])

# meo_concat = []
# for data in meo_data:
#     meo_concat.extend(data[:, -7:-1])

# for i in range(len(meo_data)):
#     print (meo_data[i][-5] > 20000).any()
#     meo_concat.extend(data[:, -7:-1])

# print (np.array(meo_concat)[:, 2] > 20000).any(), np.shape(meo_concat)
# mean_meo = np.mean(meo_concat, axis=0)
# var_meo = np.std(np.array(meo_concat) - mean_meo, axis=0)

# print mean_meo, var_meo
for data in meo_data:
    temp = data[:9504, -7:-1]
    train_meo_data.append(temp)
    temp = data[-528:, -7:-1]
    test_meo_data.append(temp)

print(mean_meo)

print(len(train_aq_data))
with tf.Session() as sess:
    model = Model(True, 0.01, 0.999)
    if tf.train.get_checkpoint_state("./bj"):
        print("LOAD BJ MODEL")
        model.saver.restore(sess, tf.train.latest_checkpoint("./bj"))
    else:
        tf.global_variables_initializer().run()

   
    ## testing step
    result = []
    temp_x = []
    temp_y = []
    for data in test_aq_data:
        temp_x.extend(np.reshape(data[-72:], (-1)))
        temp_y.extend(data[-1])
    # for data in train_meo_data:
    #     temp_x.extend(np.reshape(data[i * 24 : i * 24 + 72], (-1)))
    #     temp_y.extend(data[i * 24 + 72])  
    last_temp_x = []    
    last_predict = []          
    for k in range(48):
        feed_dict = {model.x_:[temp_x], model.y_:[temp_y], model.keep_prob:1.0}
        predict, loss = sess.run([model.y_predict, model.loss], feed_dict)
        #print len(predict[0]), len(temp)
        for ll in range(len(predict[0])):
            predict[0][ll] = predict[0][ll] * (1. + (random.random() - 0.5) * 0.1)
        result.append(predict[0])
        last_temp_x = np.array(temp_x)
        
        for s in range(35):
            temp = []
            temp = temp_x[6 + 432 * s : 432 + 432 * s]
            temp.extend(predict[0][6 * s : 6 * (s + 1)])
            temp_x[432 * s : 432 + 432 * s] = temp
        temp_y = predict[0]
        print(last_predict == np.array(predict))
        last_predict = np.array(predict)
    out = "test_id,PM2.5,PM10,O3\n"
    print(aq_station_name, len(result), len(aq_station_name),  result[0] == result[1])

    f = open("sample_submission.csv", "r").readlines()

    seq = []
    dic = {}
    for i in range(len(f)):
        if(i % 48 == 1):
            seq.append(f[i].split("_")[0])
    print(seq)
    for i in range(len(aq_station_name)):
        dic[aq_station_name[i]] = i
    
    for i in range(len(aq_station_name)):
        for j in range(48):
            out = out + seq[i] + "_aq#" + str(j) + "," + str(int(result[j][6 * dic[seq[i]] + 0])) + "," + str(int(result[j][6 * dic[seq[i]] + 1])) + "," + str(int(result[j][6 * dic[seq[i]] + 4])) + "\n"
    
    
    
    
    
    
    
    f = open("submit.csv", "w")
    f.write(out)
