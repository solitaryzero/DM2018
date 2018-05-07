import numpy as np
import pandas as pd
import os
import random
import tensorflow as tf
from model_ld import Model

aq_station_name = []
meo_station_name = []
df_aq_station = []
df_meo_station = []
for file in os.listdir("processedData/london"):
    frame = pd.read_csv("processedData/london/" + file)
    df_aq_station.append(frame)
    aq_station_name.append(file[:-4])

record_len_aq = np.shape(df_aq_station[0]["day"])[0]
keys = df_aq_station[0].keys()

dtype = []
for key in keys:
    if(key == "day" or key == "hour" or key == "year"):
        dtype.append((key, "int")) 
    else:
        dtype.append((key, "float")) 

print(aq_station_name)
print(dtype, len(keys))
print(record_len_aq)
aq_num = len(df_aq_station)

aq_data = []
for i in range(aq_num):
    temp = np.zeros([np.shape(df_aq_station[i]["day"])[0], 13])
    for j in range(len(df_aq_station[0].keys())):
        temp[:, j] = df_aq_station[i][df_aq_station[0].keys()[j]]
    aq_data.append(temp)


train_aq_data = []
test_aq_data = []
mean_aq = []
var_aq = []
for data in aq_data:
    temp = data[:-528, -6:-3]
    train_aq_data.append(temp)
    temp = data[-528:, -6:-3]
    test_aq_data.append(temp)

print(len(train_aq_data))
with tf.Session() as sess:
    model = Model(True, 0.01, 0.999)
    if tf.train.get_checkpoint_state("./ld"):
        print("LOAD LD MODEL")
        model.saver.restore(sess, tf.train.latest_checkpoint("./ld"))
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
    for k in range(48):
        feed_dict = {model.x_:[temp_x], model.y_:[temp_y], model.keep_prob:1.0}
        predict, loss = sess.run([model.y_predict, model.loss], feed_dict)
        #print len(predict[0]), len(temp)
        for ll in range(len(predict[0])):
            predict[0][ll] = predict[0][ll] * (1. + (random.random() - 0.5) * 0.3)
        result.append(predict[0])

        for s in range(13):
            temp = []
            temp = temp_x[3 + 216 * s : 216 + 216 * s]
            temp.extend(predict[0][3 * s : 3 * (s + 1)])
            temp_x[216 * s : 216 + 216 * s] = temp
    
    out = ""
    #print aq_station_name, len(result), len(aq_station_name),  result[0] == result[1]

    f = open("sample_submission.csv", "r").readlines()
    seq = []
    dic = {}
    for i in range(len(f)):
        if(i % 48 == 1):
            seq.append(f[i].split("#")[0])
    print(seq, aq_station_name)
    for i in range(len(aq_station_name)):
        dic[aq_station_name[i]] = i

    for i in range(13):
        for j in range(48):
            out = out + seq[i + 35] + "#" + str(j) + "," + str(int(result[j][3 * dic[seq[i + 35]] + 0])) + "," + str(int(result[j][3 * dic[seq[i + 35]] + 1])) + "\n"
    
    f = open("submit.csv", "r")
    temp = f.read()
    out = temp + out
    f = open("submit.csv", "w")
    f.write(out)


    # print len(correct_result)
    # score = 0
    # days = len(result) / 48.
    # for i in range(len(result)):
    #     for j in range(35):
    #         #print score, abs(result[i][6 * j + 0] - correct_result[i][6 * j + 0] + 0.), abs(result[i][6 * j + 0] + correct_result[i][6 * j + 0] + 1e-5)
    #         score += min(1., abs(result[i][6 * j + 0] - correct_result[i][6 * j + 0] + 0.) / abs(result[i][6 * j + 0] + correct_result[i][6 * j + 0] + 1e-5))
    #         score += min(1., abs(result[i][6 * j + 1] - correct_result[i][6 * j + 1] + 0.) / abs(result[i][6 * j + 1] + correct_result[i][6 * j + 1] + 1e-5))
    #         score += min(1., abs(result[i][6 * j + 4] - correct_result[i][6 * j + 4] + 0.) / abs(result[i][6 * j + 4] + correct_result[i][6 * j + 4] + 1e-5))
    # score /= 3. * len(result) * 35
    # print "TEST: ", score , len(result)
