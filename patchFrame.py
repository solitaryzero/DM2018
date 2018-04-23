import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn import datasets, linear_model

locList_aq = []
with open('location_aq.txt', 'r') as locFile:
    line = locFile.readline()
    while (line):
        lis = line.split('\t')
        locList_aq.append(lis[0])
        line = locFile.readline()

locList_me = []
with open('location_me.txt', 'r') as locFile:
    line = locFile.readline()
    while (line):
        lis = line.split(',')
        locList_me.append(lis[0])
        line = locFile.readline()

for name in locList_aq:
    filename = './processedData/'+name+'.csv'
    df = pd.read_csv(filename,header=0,index_col=0)
    df.fillna(method='pad')
    df.fillna(method='bfill')
    df.to_csv(filename)

for name in locList_me:
    filename = './processedData/'+name+'.csv'
    df = pd.read_csv(filename,header=0)
    df.fillna(method='pad')
    df.fillna(method='bfill')
    df.to_csv(filename)