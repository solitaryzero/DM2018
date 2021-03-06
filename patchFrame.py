import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn import datasets, linear_model

'''
# This program fills missing data (nan) in formatted dataFrame.
# Abnormal data(value>10000) will be abandoned too.
# When filling, use 'pad' first, 'bfill' second, and replace missing data with -1 as a last resort.
'''

locList_aq = []
with open('location_aq.txt', 'r') as locFile:
    line = locFile.readline()
    while (line):
        lis = line.split('\t')
        locList_aq.append(lis[0])
        line = locFile.readline()

print(locList_aq)

locList_me = []
with open('location_me.txt', 'r') as locFile:
    line = locFile.readline()
    while (line):
        lis = line.split(',')
        locList_me.append(lis[0])
        line = locFile.readline()

locList_aq_ld = []
with open('London_AirQuality_Stations.csv', 'r') as locFile:
    line = locFile.readline()
    line = locFile.readline()
    while (line):
        lis = line.split(',')
        if (lis[2] == "TRUE"):
            locList_aq_ld.append(lis[0])
        line = locFile.readline()

for name in locList_aq:
    filename = './processedData/beijing/'+name+'.csv'
    df = pd.read_csv(filename,header=0,index_col=0)
    df = df[df < 10000]
    df = df.interpolate()
    df.fillna(method='pad',inplace=True)
    df.fillna(method='bfill',inplace=True)
    df.fillna(-1,inplace=True)
    df.to_csv(filename)

for name in locList_me:
    filename = './processedData/beijing/'+name+'.csv'
    df = pd.read_csv(filename,header=0,index_col=0)
    df = df[df < 10000]
    df = df.interpolate()
    df.fillna(method='pad',inplace=True)
    df.fillna(method='bfill',inplace=True)
    df.fillna(-1,inplace=True)
    df.to_csv(filename)

for name in locList_aq_ld:
    filename = './processedData/london/'+name+'.csv'
    df = pd.read_csv(filename,header=0,index_col=0)
    df = df[df < 10000]
    df = df.interpolate()
    df.fillna(method='pad',inplace=True)
    df.fillna(method='bfill',inplace=True)
    df.fillna(-1,inplace=True)
    df.to_csv(filename)