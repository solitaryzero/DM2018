import numpy as np
import pandas as pd
import os
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

print meo_station_name
print dtype, len(keys)
print record_len_aq, record_len_meo
aq_num = len(df_aq_station)
meo_num = len(df_meo_station)

aq_data = []
meo_data = []
for i in range(aq_num):
    temp = np.zeros([record_len_aq, 13])
    for j in range(len(df_aq_station[0].keys())):
        temp[:, j] = df_aq_station[i][df_aq_station[0].keys()[j]]
    aq_data.append(temp)

for i in range(meo_num):
    print i
    temp = np.zeros([record_len_meo, 14])
    for j in range(len(df_meo_station[0].keys())):
        temp[:, j] = df_meo_station[i][df_meo_station[0].keys()[j]]
    meo_data.append(temp)


train_aq_data = []
test_aq_data = []
mean_aq = []
var_aq = []
for data in aq_data:
    temp = data[:9504, -6:]
    train_aq_data.append(temp)
    temp = data[-528:, -6:]
    test_aq_data.append(temp)

train_meo_data = []
test_meo_data = []
mean_meo = np.zeros([6])
var_meo = np.zeros([6])

meo_concat = []
for data in meo_data:
    meo_concat.extend(data[:, -7:-1])

for i in range(len(meo_data)):
    print (meo_data[i][-5] > 20000).any()
    meo_concat.extend(data[:, -7:-1])

print (np.array(meo_concat)[:, 2] > 20000).any(), np.shape(meo_concat)
mean_meo = np.mean(meo_concat, axis=0)
var_meo = np.std(np.array(meo_concat) - mean_meo, axis=0)

print mean_meo, var_meo
for data in meo_data:
    temp = data[:9504, -7:-1]
    train_meo_data.append(temp)
    temp = data[-528:, -7:-1]
    test_meo_data.append(temp)

print mean_meo

print len(train_aq_data)
with tf.Session() as sess:
    model = Model(True, 0.01, 0.999)
    tf.global_variables_initializer().run()
   
    for epoch in range(10):
        index = 0
        print len(train_aq_data[0]) - 73
        ## training step
        while(index < len(train_aq_data[0]) - 73):
            #print index
            x_in = []
            y_in = []
            for data in train_aq_data:
                x_in.extend(np.reshape(data[index : index + 72], (-1)))
                y_in.extend(data[index + 72])
                #], data[index + 72, 1], data[index + 72, 4]])
            for data in train_meo_data:
                x_in.extend(np.reshape(data[index : index + 72], (-1)))
                y_in.extend(data[index + 72])
            feed_dict = {model.x_:[x_in], model.y_:[y_in], model.keep_prob:0.5}
            predict, loss, _ = sess.run([model.y_predict, model.loss, model.train_op], feed_dict)
            if(index % 100 == 0):
                print index, loss
            index += 1

        model.saver.save(sess, 'bj/checkpoint', global_step=epoch)

        ## testing step
        result = []
        correct_result = []
        for i in range(17):
            temp_x = []
            temp_y = []
            for data in test_aq_data:
                temp_x.extend(np.reshape(data[i * 24 : i * 24 + 72], (-1)))
                temp_y.extend(data[i * 24 + 72])
            for data in train_meo_data:
                temp_x.extend(np.reshape(data[i * 24 : i * 24 + 72], (-1)))
                temp_y.extend(data[i * 24 + 72])                
            for k in range(48):
                feed_dict = {model.x_:[temp_x], model.y_:[temp_y], model.keep_prob:1.0}
                predict, loss = sess.run([model.y_predict, model.loss], feed_dict)
                #print len(predict[0]), len(temp)
                result.append(predict[0])
                correct_result.append(temp_y)

                for s in range(53):
                    temp = []
                    temp = temp_x[6 + 432 * s : 432 + 432 * s]
                    temp.extend(predict[0][6 * s : 6 * (s + 1)])
                    temp_x[432 * s : 432 + 432 * s] = temp

                temp_y = []
                #print i, k
                for data in test_aq_data:
                    temp_y.extend(data[i * 24 + 72 + k + 1])
                for data in train_meo_data:
                    temp_y.extend(data[i * 24 + 72 + k + 1])  
        
        print len(correct_result)
        score = 0
        days = len(result) / 48.
        #print len(result[0]), days
        for i in range(len(result)):
            #print i, result[i]
            for j in range(35):
                #print score, abs(result[i][6 * j + 0] - correct_result[i][6 * j + 0] + 0.), abs(result[i][6 * j + 0] + correct_result[i][6 * j + 0] + 1e-5)
                score += min(1., abs(result[i][6 * j + 0] - correct_result[i][6 * j + 0] + 0.) / abs(result[i][6 * j + 0] + correct_result[i][6 * j + 0] + 1e-5))
                score += min(1., abs(result[i][6 * j + 1] - correct_result[i][6 * j + 1] + 0.) / abs(result[i][6 * j + 1] + correct_result[i][6 * j + 1] + 1e-5))
                score += min(1., abs(result[i][6 * j + 4] - correct_result[i][6 * j + 4] + 0.) / abs(result[i][6 * j + 4] + correct_result[i][6 * j + 4] + 1e-5))
        score /= 3. * len(result) * 35
        print "TEST: ", score , len(result)
