#encoding=utf-8
import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn import metrics

def readData_aq_bj(name):
    df = pd.read_csv(filepath_or_buffer='./processedData/beijing/' + name + '.csv', header=0, index_col=0)
    return df

def readData_aq_ld(name):
    df = pd.read_csv(filepath_or_buffer='./processedData/london/' + name + '.csv', header=0, index_col=0)
    return df

def writeRes_bj(resFile,siteName,i,Y_output):
    resFile.write(siteName + '#' + str(i) + ',')
    if (Y_output[0][0] < 0):
        resFile.write('0.0')
    else:
        resFile.write(str(Y_output[0][0]))
    resFile.write(',')
    if (Y_output[0][1] < 0):
        resFile.write('0.0')
    else:
        resFile.write(str(Y_output[0][1]))
    resFile.write(',')
    if (Y_output[0][2] < 0):
        resFile.write('0.0')
    else:
        resFile.write(str(Y_output[0][2]))
    resFile.write('\n')

def writeRes_ld(resFile,siteName,i,Y_output):
    resFile.write(siteName + '#' + str(i) + ',')
    if (Y_output[0][0] < 0):
        resFile.write('0.0')
    else:
        resFile.write(str(Y_output[0][0]))
    resFile.write(',')
    if (Y_output[0][1] < 0):
        resFile.write('0.0')
    else:
        resFile.write(str(Y_output[0][1]))
    resFile.write(',\n')

def linearReg_bj(siteName):
    print(siteName)
    res_aq = readData_aq_bj(siteName)

    # res_aq = res_aq.loc[:,['PM25_Concentration','PM10_Concentration','NO2_Concentration','CO_Concentration','O3_Concentration','SO2_Concentration']]
    res_aq = res_aq.loc[:, ['PM25_Concentration', 'PM10_Concentration', 'O3_Concentration']]

    res_aq_mat = res_aq.as_matrix()
    res_rows = res_aq.shape[0]

    variables = np.zeros((res_rows - dayRange * 24, pollutionType * dayRange * 24))
    results = np.zeros((res_rows - dayRange * 24, pollutionType))

    for i in range(0, res_rows - dayRange * 24):
        for j in range(0, dayRange * 24):
            for k in range(0, pollutionType):
                variables[i][j * pollutionType + k] = res_aq_mat[i + j][k]

    for i in range(0, res_rows - dayRange * 24):
        for k in range(0, pollutionType):
            results[i][k] = res_aq_mat[i + dayRange * 24][k]

    #X_train, X_test, y_train, y_test = train_test_split(variables, results, random_state=1)
    linreg = LinearRegression()
    linreg.fit(variables, results)
    # 模型拟合测试集
    #y_pred = linreg.predict(X_test)
    # 用scikit-learn计算MSE
    #print("MSE:", metrics.mean_squared_error(y_test, y_pred))
    # 用scikit-learn计算RMSE
    #print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    #print(linreg.intercept_)
    #print(linreg.coef_)

    X_input = np.zeros((48, pollutionType * dayRange * 24))
    Y_output = np.zeros((1, pollutionType))
    for i in range(0, 48):
        if (i == 0):
            for j in range(0, dayRange * 24):
                for k in range(0, pollutionType):
                    X_input[i][j * pollutionType + k] = res_aq_mat[res_rows - dayRange * 24 + j][k]
        else:
            for j in range(0, (dayRange * 24 - 1) * pollutionType):
                X_input[i][j] = X_input[i - 1][j + pollutionType]
            for j in range(0, pollutionType):
                X_input[i][(dayRange * 24 - 1) * pollutionType + j] = Y_output[0][j]

        final_input = X_input[i]
        final_input = final_input.reshape(1, pollutionType * dayRange * 24)
        Y_output = linreg.predict(final_input)
        writeRes_bj(resFile,siteName,i,Y_output)

def linearReg_ld(siteName):
    print(siteName)
    res_aq = readData_aq_ld(siteName)

    # res_aq = res_aq.loc[:,['PM25_Concentration','PM10_Concentration','NO2_Concentration','CO_Concentration','O3_Concentration','SO2_Concentration']]
    res_aq = res_aq.loc[:, ['PM25_Concentration', 'PM10_Concentration', 'O3_Concentration']]

    res_aq_mat = res_aq.as_matrix()
    res_rows = res_aq.shape[0]

    variables = np.zeros((res_rows - dayRange * 24, pollutionType * dayRange * 24))
    results = np.zeros((res_rows - dayRange * 24, pollutionType))

    for i in range(0, res_rows - dayRange * 24):
        for j in range(0, dayRange * 24):
            for k in range(0, pollutionType):
                variables[i][j * pollutionType + k] = res_aq_mat[i + j][k]

    for i in range(0, res_rows - dayRange * 24):
        for k in range(0, pollutionType):
            results[i][k] = res_aq_mat[i + dayRange * 24][k]

    #X_train, X_test, y_train, y_test = train_test_split(variables, results, random_state=1)
    linreg = LinearRegression()
    linreg.fit(variables, results)
    # 模型拟合测试集
    #y_pred = linreg.predict(X_test)
    # 用scikit-learn计算MSE
    #print("MSE:", metrics.mean_squared_error(y_test, y_pred))
    # 用scikit-learn计算RMSE
    #print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    X_input = np.zeros((48, pollutionType * dayRange * 24))
    Y_output = np.zeros((1, pollutionType))
    for i in range(0, 48):
        if (i == 0):
            for j in range(0, dayRange * 24):
                for k in range(0, pollutionType):
                    X_input[i][j * pollutionType + k] = res_aq_mat[res_rows - dayRange * 24 + j][k]
        else:
            for j in range(0, (dayRange * 24 - 1) * pollutionType):
                X_input[i][j] = X_input[i - 1][j + pollutionType]
            for j in range(0, pollutionType):
                X_input[i][(dayRange * 24 - 1) * pollutionType + j] = Y_output[0][j]

        final_input = X_input[i]
        final_input = final_input.reshape(1, pollutionType * dayRange * 24)
        Y_output = linreg.predict(final_input)
        writeRes_ld(resFile,siteName,i,Y_output)

stationNum_aq = 35
stationNum_me = 18
dayRange = 5
pollutionType = 3
resFileName = "linearReg_submission.csv"
resFile = open(resFileName,'w')
resFile.write('test_id,PM2.5,PM10,O3\n')

locList_aq_bj = []
with open('location_aq.txt','r') as locFile:
    line = locFile.readline()
    while (line):
        lis = line.split('\t')
        locList_aq_bj.append(lis[0])
        line = locFile.readline()

for i in range(0,len(locList_aq_bj)):
    linearReg_bj(locList_aq_bj[i])

locList_aq_ld = []
with open('location_aq_ld.txt', 'r') as locFile:
    line = locFile.readline()
    while (line):
        lis = line.strip().split('\t')
        locList_aq_ld.append(lis[0])
        line = locFile.readline()

for i in range(0,len(locList_aq_ld)):
    linearReg_ld(locList_aq_ld[i])

resFile.close()
