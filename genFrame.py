import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn import datasets, linear_model

def getFileName(city,type,date):
    date_str = str(date)
    if (date<10):
        date_str = '0'+date_str
    return './data/'+city+'_'+type+'_'+'2018-04-'+date_str+'-0-'+'2018-04-'+date_str+'-23.csv'

def genDataFrame(city,type,startDate,endDate):
    df = pd.read_csv(filepath_or_buffer=getFileName(city, type, startDate), header=0,index_col='id')
    for i in range(startDate+1,endDate+1):
        newdf = pd.read_csv(filepath_or_buffer=getFileName(city, type, i), header=0,index_col='id')
        df = pd.concat([df,newdf],ignore_index=False)
    return df

def genTimeString(date,hour):
    date_str = str(date)
    if (date<10):
        date_str = '0'+date_str
    hour_str = str(hour)
    if (hour<10):
        hour_str = '0'+hour_str
    return '2018-04-'+date_str+' '+hour_str+':00:00'

def getByTime(df,date,hour):
    return df[df.time == genTimeString(date,hour)]

siteNum_aq = 35
siteNum_me = 18
startDate = 1
endDate = 20
df_aq = genDataFrame('bj','airquality',startDate,endDate)
df_me = genDataFrame('bj','meteorology',startDate,endDate)
stationNames = df_aq['station_id'].unique()
dfs_station = []
for i in range(0,len(stationNames)):
    df = df_aq[df_aq.station_id == stationNames[i]]
    df = df.fillna(method='pad')
    df = df.fillna(method='bfill')
    dfs_station.append(df)

print(dfs_station[0])