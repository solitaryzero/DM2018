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
    aq_station_name.append(file)

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
    print(i, record_len_aq)
    temp = np.zeros([np.shape(df_aq_station[i]["day"])[0], 13])
    for j in range(len(df_aq_station[0].keys())):
        print(df_aq_station[0].keys()[j], len(df_aq_station[i][df_aq_station[0].keys()[j]]))
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
   
    for epoch in range(5):
        index = 0
        print(len(train_aq_data[0]) - 73)
        ## training step
        while(index < len(train_aq_data[0]) - 73):
            #print(index)
            x_in = []
            y_in = []
            for data in train_aq_data:
                x_in.extend(np.reshape(data[index : index + 72], (-1)))
                y_in.extend(data[index + 72])
                #], data[index + 72, 1], data[index + 72, 4]])
            feed_dict = {model.x_:[x_in], model.y_:[y_in], model.keep_prob:0.5}
            predict, loss, _ = sess.run([model.y_predict, model.loss, model.train_op], feed_dict)
            if(index % 100 == 0):
                print(index, loss)
            index += 1
        
        model.saver.save(sess, 'ld/checkpoint', global_step=epoch)

        ## testing step
        result = []
        correct_result = []
        for i in range(17):
            temp_x = []
            temp_y = []
            for data in test_aq_data:
                temp_x.extend(np.reshape(data[i * 24 : i * 24 + 72], (-1)))
                temp_y.extend(data[i * 24 + 72])               
            for k in range(48):
                #print(np.shape(temp_x))
                feed_dict = {model.x_:[temp_x], model.y_:[temp_y], model.keep_prob:1.0}
                predict, loss = sess.run([model.y_predict, model.loss], feed_dict)
                for ll in range(len(predict[0])):
                    predict[0][ll] = predict[0][ll] * (1. + (random.random() - 0.5) * 0.2)
                #print(len(predict[0]), len(temp))
                result.append(predict[0])
                correct_result.append(temp_y)

                for s in range(13):
                    temp = []
                    temp = temp_x[3 + 216 * s : 216 + 216 * s]
                    temp.extend(predict[0][3 * s : 3 * (s + 1)])
                    temp_x[216 * s : 216 + 216 * s] = temp

                temp_y = []
                #print(i, k)
                for data in test_aq_data:
                    temp_y.extend(data[i * 24 + 72 + k + 1])

        
        print(len(correct_result))
        score = 0
        days = len(result) / 48
        #print(len(result[0]), days)
        for i in range(len(result)):
            #print(i, result[i])
            for j in range(13):
                #print(score, abs(result[i][6 * j + 0] - correct_result[i][6 * j + 0] + 0.), abs(result[i][6 * j + 0] + correct_result[i][6 * j + 0] + 1e-5))
                score += min(1., abs(result[i][3 * j + 0] - correct_result[i][3 * j + 0] + 0.) / abs(result[i][3 * j + 0] + correct_result[i][3 * j + 0] + 1e-5))
                score += min(1., abs(result[i][3 * j + 1] - correct_result[i][3 * j + 1] + 0.) / abs(result[i][3 * j + 1] + correct_result[i][3 * j + 1] + 1e-5))
        score /= 2. * len(result) * 13
        print("TEST: ", score , len(result))
        np.savetxt("result.txt", np.maximum(0, np.int16(result)))
