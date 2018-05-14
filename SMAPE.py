import numpy as np
import pandas as pd

def smape(actual, predicted):
    a = np.abs(np.array(actual) - np.array(predicted))
    b = np.array(actual) + np.array(predicted)

    return 2 * np.mean(np.divide(a, b, out=np.zeros_like(a)))

date = int(input("date: "))
lassoFile = open("lasso_submission.csv",'r')
linFile = open("linearReg_submission.csv",'r')
actFile = open('dongsi_test.csv','r')

df = pd.read_csv(actFile)
actual = df.loc[:, ['PM25_Concentration', 'PM10_Concentration', 'O3_Concentration']].as_matrix()

df = pd.read_csv(linFile)
predicted = df.loc[0:47, ['PM2.5','PM10','O3']].as_matrix()
print("SMAPE for linear: ", str(smape(actual,predicted)))

df = pd.read_csv(lassoFile)
predicted = df.loc[0:47, ['PM2.5','PM10','O3']].as_matrix()
print("SMAPE for lasso: ", str(smape(actual,predicted)))

