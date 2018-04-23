import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import genFrame as gf
from sklearn import metrics

stationNum_aq = 34
stationNum_me = 18
dayRange = 5
resFileName = "linearReg_submission.csv"
resFile = open(resFileName,'w')
resFile.write('test_id,PM2.5,PM10,O3\n')

locList_aq = []
with open('location_aq.txt','r') as locFile:
    line = locFile.readline()
    while (line):
        lis = line.split('\t')
        locList_aq.append(lis[0])
        line = locFile.readline()

startDate = 1
endDate = 22
res = gf.genFrames(startDate,endDate)
res_aq = res[0]
res_me = res[1]
for i in range(0,stationNum_aq):
    print(str(i)+' '+str(res_aq[i].isnull().values.any()))

variables = np.zeros(((endDate-startDate-dayRange)*stationNum_aq,dayRange*24*6))
results = np.zeros(((endDate-startDate-dayRange)*stationNum_aq,48*6))
dayLength = endDate-startDate-dayRange
for p in range(0,stationNum_aq):
    res_m_aq = res_aq[p].as_matrix()
    for i in range(startDate+dayRange,endDate):
        for j in range(0, dayRange):
            for k in range(0,24):
                for l in range(0,6):
                    variables[dayLength*p+(i-startDate-dayRange)][j*24*6+k*6+l] = \
                        res_m_aq[(i-startDate-dayRange+j)*24+k][l+4]
                    results[dayLength*p+(i-startDate-dayRange)][k*6+l] = \
                        res_m_aq[(i-startDate)*24+k][l+4]
                    results[dayLength*p+(i-startDate-dayRange)][(24+k)*6+l] = \
                        res_m_aq[(i-startDate+1)*24+k][l+4]

X_train, X_test, y_train, y_test = train_test_split(variables, results, random_state=1)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
linreg = LinearRegression()
linreg.fit(X_train, y_train)
#模型拟合测试集
y_pred = linreg.predict(X_test)
# 用scikit-learn计算MSE
print("MSE:",metrics.mean_squared_error(y_test, y_pred))
# 用scikit-learn计算RMSE
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

X_input = np.zeros((stationNum_aq,dayRange*24*6))
for p in range(0,stationNum_aq):
    res_m_aq = res_aq[p].as_matrix()
    for j in range(0, dayRange):
        for k in range(0,24):
            for l in range(0,6):
                X_input[p][j*24*6+k*6+l] = \
                    res_m_aq[(endDate-startDate-dayRange+j)*24+k][l+4]

Y_output = linreg.predict(X_input)
for p in range(0,stationNum_aq):
    if (p < 8):
        name_str = locList_aq[p]
    else:
        name_str = locList_aq[p+1]
    for i in range(0,48):
        resFile.write(name_str+'#'+str(i)+','+str(Y_output[p][4])+','+str(Y_output[p][5])+','+str(Y_output[p][8])+'\n')

resFile.close()