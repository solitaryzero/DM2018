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

def genFrames(startDate,endDate):
    siteNum_aq = 35
    siteNum_me = 18
    df_aq = genDataFrame('bj','airquality',startDate,endDate)
    df_me = genDataFrame('bj','meteorology',startDate,endDate)

    stationNames_aq = df_aq['station_id'].unique()
    df_aq_station = []
    for i in range(0,len(stationNames_aq)):
        df = df_aq[df_aq.station_id == stationNames_aq[i]]
        df = df.fillna(method='pad')
        df = df.fillna(method='bfill')
        df_aq_station.append(df)
    #print(df_aq_station[0])

    stationNames_me = df_me['station_id'].unique()
    df_me_station = []
    for i in range(0,len(stationNames_me)):
        df = df_me[df_me.station_id == stationNames_me[i]]
        df = df.fillna(method='pad')
        df = df.fillna(method='bfill')
        df_me_station.append(df)
    #print(df_me_station[0])

    return df_aq_station,df_me_station

res = genFrames(1,20)
res_aq = res[0]
res_me = res[1]
print(res_aq[0])
print(res_me[0])