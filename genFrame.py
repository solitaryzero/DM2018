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
    hour_str = str(hour)
    if (hour<10):
        hour_str = '0'+hour_str
    if (date > 30):
        date_str = str(date-30)
        if (date < 40):
            date_str = '0' + date_str
        return '2018-05-' + date_str + ' ' + hour_str + ':00:00'
    else:
        date_str = str(date)
        if (date < 10):
            date_str = '0' + date_str
        return '2018-04-'+date_str+' '+hour_str+':00:00'

def getByTime(df,date,hour):
    return df[df.time == genTimeString(date,hour)]

def genFrames(startDate,endDate):
    siteNum_aq = 35
    siteNum_me = 18

    locDict_aq = {}
    with open('location_aq.txt','r') as locFile:
        line = locFile.readline()
        while (line):
            lis = line.split('\t')
            locDict_aq[lis[0]] = [float(lis[1]),float(lis[2])]
            line = locFile.readline()

    locDict_me = {}
    with open('location_me.txt','r') as locFile:
        line = locFile.readline()
        while (line):
            lis = line.split(',')
            locDict_me[lis[0]] = [float(lis[1]),float(lis[2])]
            line = locFile.readline()

    df_aq = genDataFrame('bj','airquality',startDate,endDate)
    df_me = genDataFrame('bj','meteorology',startDate,endDate)

    stationNames_aq = df_aq['station_id'].unique()
    df_aq_station = []
    for i in range(0,len(stationNames_aq)):
        if (stationNames_aq[i] == 'zhiwuyuan_aq'):
            continue
        df = df_aq[df_aq.station_id == stationNames_aq[i]]
        df = df.fillna(method='pad')
        df = df.fillna(method='bfill')
        tmp = np.zeros(((endDate-startDate+1)*24,10))
        siteLoc = locDict_aq[stationNames_aq[i]]
        for j in range(0,(endDate-startDate+1)*24):
            tmp[j][0] = siteLoc[0]  #经度
            tmp[j][1] = siteLoc[1]  #纬度
            tmp[j][2] = int(j/24)   #日期（距04-01距离）
            tmp[j][3] = j%24        #小时
            date = int(j/24)+1
            hour = j%24
            forward = False
            while (not(genTimeString(date, hour) in df.time.values)):
                if (forward):
                    hour += 1
                    if (hour == 24):
                        date += 1
                        hour = 0
                else:
                    hour -= 1
                    if (hour == -1):
                        date -= 1
                        hour = 23
                if (date == 0):
                    date = int(j/24)+1
                    hour = j%24
                    forward = True

            datLine = df[df.time == genTimeString(date, hour)]
            tmp[j][4] = datLine['PM25_Concentration']
            tmp[j][5] = datLine['PM10_Concentration']
            tmp[j][6] = datLine['NO2_Concentration']
            tmp[j][7] = datLine['CO_Concentration']
            tmp[j][8] = datLine['O3_Concentration']
            tmp[j][9] = datLine['SO2_Concentration']
        newdf = pd.DataFrame(tmp,columns=['longitude','latitude','day','hour','PM25_Concentration',
                                          'PM10_Concentration','NO2_Concentration','CO_Concentration',
                                          'O3_Concentration','SO2_Concentration'])
        df_aq_station.append(newdf)
    #print(df_aq_station[0])

    stationNames_me = df_me['station_id'].unique()
    df_me_station = []
    weatherList = ['Sunny/clear','Hail','Thundershower','Sleet','Cloudy','Light Rain','Overcast','Rain']
    for i in range(0,len(stationNames_me)):
        df = df_me[df_me.station_id == stationNames_me[i]]
        df = df.fillna(method='pad')
        df = df.fillna(method='bfill')
        tmp = np.zeros(((endDate - startDate + 1) * 24, 10))
        siteLoc = locDict_me[stationNames_me[i]]
        for j in range(0, (endDate - startDate + 1) * 24):
            tmp[j][0] = siteLoc[0]  # 经度
            tmp[j][1] = siteLoc[1]  # 纬度
            tmp[j][2] = int(j / 24)  # 日期（距04-01距离）
            tmp[j][3] = j % 24  # 小时
            date = int(j / 24) + 1
            hour = j % 24
            forward = False
            while (not (genTimeString(date, hour) in df.time.values)):
                if (forward):
                    hour += 1
                    if (hour == 24):
                        date += 1
                        hour = 0
                else:
                    hour -= 1
                    if (hour == -1):
                        date -= 1
                        hour = 23
                if (date == 0):
                    date = int(j / 24) + 1
                    hour = j % 24
                    forward = True

            datLine = df[df.time == genTimeString(date, hour)]
            tmp[j][4] = weatherList.index(datLine['weather'].values) #weather
            tmp[j][5] = datLine['temperature']
            tmp[j][6] = datLine['pressure']
            tmp[j][7] = datLine['humidity']
            tmp[j][8] = datLine['wind_direction']
            tmp[j][9] = datLine['wind_speed']
        newdf = pd.DataFrame(tmp, columns=['longitude', 'latitude', 'day', 'hour', 'weather',
                                           'temperature', 'pressure', 'humidity',
                                           'wind_direction', 'wind_speed'])
        df_me_station.append(newdf)
    #print(df_me_station[0])

    return df_aq_station,df_me_station

'''
res = genFrames(1,20)
res_aq = res[0]
res_me = res[1]
print(res_aq[0])
print(res_me[0])
'''