import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn import datasets, linear_model


def getFileName(city, type, day):
    month_str = '04'
    if (day > 30):
        month_str = '05'
        day = day - 30

    day_str = str(day)
    if (day < 10):
        day_str = '0'+day_str

    return './data/' + city + '_' + type + '_' + '2018-'+ month_str + '-' + day_str + '-0-' + '2018-'+ month_str + '-' + day_str + '-23.csv'

def genDataFrame(city, type, startDate, endDate):
    df = pd.read_csv(filepath_or_buffer=getFileName(city, type, startDate), header=0, index_col='id')
    for i in range(startDate + 1, endDate + 1):
        newdf = pd.read_csv(filepath_or_buffer=getFileName(city, type, i), header=0, index_col='id')
        df = pd.concat([df, newdf], ignore_index=False)
    return df


def genDataFrame_old_aq_bj():
    df = pd.read_csv(filepath_or_buffer='beijing_17_18_aq.csv', header=0)
    df2 = pd.read_csv(filepath_or_buffer='beijing_201802_201803_aq.csv', header=0)
    df = pd.concat([df, df2], ignore_index=False)
    return df


def genDataFrame_old_aq_ld():
    df = pd.read_csv(filepath_or_buffer='London_historical_aqi_forecast_stations_20180331.csv', header=0, index_col=0)
    df2 = pd.read_csv(filepath_or_buffer='London_historical_aqi_other_stations_20180331.csv', header=0)
    df = pd.concat([df, df2], ignore_index=True)
    return df


def genDataFrame_old_me_bj():
    df = pd.read_csv(filepath_or_buffer='beijing_17_18_meo.csv', header=0)
    return df


def genTimeString(year, month, day, hour):
    hour_str = str(hour)
    if (hour < 10):
        hour_str = '0' + hour_str
    day_str = str(day)
    if (day < 10):
        day_str = '0' + day_str
    month_str = str(month)
    if (month < 10):
        month_str = '0' + month_str
    year_str = str(year)
    return year_str + '-' + month_str + '-' + day_str + ' ' + hour_str + ':00:00'


def genTimeString_ld_old(year, month, day, hour):
    hour_str = str(hour)
    day_str = str(day)
    month_str = str(month)
    year_str = str(year)
    return year_str + '/' + month_str + '/' + day_str + ' ' + hour_str + ':00'


def genFrames_bj(startDate, endDate, save):
    siteNum_aq = 35
    siteNum_me = 18

    locDict_aq = {}
    with open('location_aq.txt', 'r') as locFile:
        line = locFile.readline()
        while (line):
            lis = line.split('\t')
            locDict_aq[lis[0]] = [float(lis[1]), float(lis[2])]
            line = locFile.readline()

    locDict_me = {}
    with open('location_me.txt', 'r') as locFile:
        line = locFile.readline()
        while (line):
            lis = line.split(',')
            locDict_me[lis[0]] = [float(lis[1]), float(lis[2])]
            line = locFile.readline()

    df_aq = genDataFrame('bj', 'airquality', startDate, endDate)
    df_me = genDataFrame('bj', 'meteorology', startDate, endDate)

    monthLength = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


    stationNames_aq = df_aq['station_id'].unique()
    columnList_aq = ['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'PM25_Concentration',
                     'PM10_Concentration', 'NO2_Concentration', 'CO_Concentration',
                     'O3_Concentration', 'SO2_Concentration']
    df_aq_station = []

    for i in range(0, len(stationNames_aq)):
        df = df_aq[df_aq.station_id == stationNames_aq[i]]
        df = df.fillna(method='pad')
        df = df.fillna(method='bfill')

        if (stationNames_aq[i] == 'zhiwuyuan_aq'):
            df_backup = df_aq[df_aq.station_id == stationNames_aq[i - 1]]

        tmp = np.zeros(((endDate - startDate + 1) * 24, 12))
        siteLoc = locDict_aq[stationNames_aq[i]]
        for j in range(0, (endDate - startDate + 1) * 24):
            tmp[j][0] = siteLoc[0]  # 经度
            tmp[j][1] = siteLoc[1]  # 纬度
            tmp[j][2] = 2018  # 年
            tmp[j][3] = int(((j / 24)+startDate-1) / 30) + 4  # 月
            tmp[j][4] = (int(j / 24) % 30 + startDate - 1) % 30 + 1  # 日
            tmp[j][5] = j % 24  # 小时
            day = (int(j / 24) % 30 + startDate - 1) % 30 + 1
            month = int(((j / 24)+startDate-1) / 30) + 4
            hour = j % 24
            forward = False
            flag = True
            timeSet = df.time.values
            if (stationNames_aq[i] == 'zhiwuyuan_aq'):
                flag = False
                timeSet = df_backup.time.values

            while not (genTimeString(2018, month, day, hour) in timeSet):
                if (forward):
                    hour += 1
                    if (hour == 24):
                        day += 1
                        hour = 0
                        if (day > monthLength[month - 1]):
                            day = 1
                            month += 1
                else:
                    hour -= 1
                    if (hour == -1):
                        day -= 1
                        hour = 23
                        if (day == 0):
                            month -= 1
                            day = monthLength[month - 1]
                if (month == 3):
                    day = (int(j / 24) % 30 + startDate - 1) % 30 + 1
                    month = int(((j / 24)+startDate-1) / 30) + 4
                    hour = j % 24
                    forward = True

            if (flag):
                datLine = df[df.time == genTimeString(2018, month, day, hour)]
            else:
                datLine = df_backup[df_backup.time == genTimeString(2018, month, day, hour)]
            if (len(datLine) > 1):
                datLine = datLine.drop_duplicates()
            # print(datLine)

            tmp[j][6] = datLine['PM25_Concentration']
            tmp[j][7] = datLine['PM10_Concentration']
            tmp[j][8] = datLine['NO2_Concentration']
            tmp[j][9] = datLine['CO_Concentration']
            tmp[j][10] = datLine['O3_Concentration']
            tmp[j][11] = datLine['SO2_Concentration']
        newdf = pd.DataFrame(tmp, columns=columnList_aq)
        if (save):
            olddf = pd.read_csv('./processedData/beijing/' + stationNames_aq[i] + '.csv', header=0, index_col=0)
            fulldf = pd.concat([olddf,newdf], ignore_index=True)
            fulldf.to_csv('./processedData/beijing/' + stationNames_aq[i] + '.csv')
        df_aq_station.append(newdf)
    # print(df_aq_station[0])

    stationNames_me = df_me['station_id'].unique()
    df_me_station = []
    weatherList = ['Sunny/clear', 'Hail', 'Thundershower', 'Sleet', 'Cloudy', 'Light Rain', 'Overcast', 'Rain', 'Fog',
                   'Snow', 'Haze', 'Dust', 'Sand', 'Rain/Snow with Hail', 'Rain with Hail']
    columnList_me = ['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'weather',
                     'temperature', 'pressure', 'humidity',
                     'wind_direction', 'wind_speed']

    for i in range(0, len(stationNames_me)):
        df = df_me[df_me.station_id == stationNames_me[i]]
        df = df.fillna(method='pad')
        df = df.fillna(method='bfill')

        tmp = np.zeros(((endDate - startDate + 1) * 24, 12))
        siteLoc = locDict_me[stationNames_me[i]]
        for j in range(0, (endDate - startDate + 1) * 24):
            tmp[j][0] = siteLoc[0]  # 经度
            tmp[j][1] = siteLoc[1]  # 纬度
            tmp[j][2] = 2018  # 年
            tmp[j][3] = int(((j / 24)+startDate-1) / 30) + 4  # 月
            tmp[j][4] = (int(j / 24) % 30 + startDate - 1) % 30 + 1  # 日
            tmp[j][5] = j % 24  # 小时
            day = (int(j / 24) % 30 + startDate - 1) % 30 + 1
            month = int(((j / 24)+startDate-1) / 30) + 4
            hour = j % 24
            forward = False
            while not (genTimeString(2018, month, day, hour) in df.time.values):
                if (forward):
                    hour += 1
                    if (hour == 24):
                        day += 1
                        hour = 0
                        if (day > monthLength[month - 1]):
                            day = 1
                            month += 1
                else:
                    hour -= 1
                    if (hour == -1):
                        day -= 1
                        hour = 23
                        if (day == 0):
                            month -= 1
                            day = monthLength[month - 1]
                if (month == 3):
                    day = (int(j / 24) % 30 + startDate - 1) % 30 + 1
                    month = int(((j / 24)+startDate-1) / 30) + 4
                    hour = j % 24
                    forward = True

            datLine = df[df.time == genTimeString(2018, month, day, hour)].head(1)
            tmp[j][6] = weatherList.index(datLine['weather'].values)
            tmp[j][7] = datLine['temperature']
            tmp[j][8] = datLine['pressure']
            tmp[j][9] = datLine['humidity']
            tmp[j][10] = datLine['wind_direction']
            tmp[j][11] = datLine['wind_speed']
        newdf = pd.DataFrame(tmp, columns=columnList_me)
        if (save):
            olddf = pd.read_csv('./processedData/beijing/' + stationNames_me[i] + '.csv', header=0, index_col=0)
            fulldf = pd.concat([olddf, newdf], ignore_index=True)
            fulldf.to_csv('./processedData/beijing/' + stationNames_me[i] + '.csv')
        df_me_station.append(newdf)
    print(df_me_station[0])

    return df_aq_station, df_me_station


def genFrames_ld(startDate, endDate, save):  # TODO
    locDict_aq = {}
    with open('London_AirQuality_Stations.csv', 'r') as locFile:
        line = locFile.readline()
        line = locFile.readline()
        while (line):
            lis = line.split(',')
            if (lis[2] == "TRUE"):
                locDict_aq[lis[0]] = [float(lis[4]), float(lis[5])]
            line = locFile.readline()

    df_aq = genDataFrame('ld', 'airquality', startDate, endDate)

    monthLength = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    stationNames_aq = locDict_aq.keys()
    columnList_aq = ['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'PM25_Concentration',
                     'PM10_Concentration', 'NO2_Concentration', 'CO_Concentration',
                     'O3_Concentration', 'SO2_Concentration']
    df_aq_station = []
    for name in stationNames_aq:
        df = df_aq[df_aq.station_id == name]
        df = df.fillna(method='pad')
        df = df.fillna(method='bfill')
        df = df.fillna(-1)

        tmp = np.zeros(((endDate - startDate + 1) * 24, 12))
        siteLoc = locDict_aq[name]
        for j in range(0, (endDate - startDate + 1) * 24):
            tmp[j][0] = siteLoc[0]  # 经度
            tmp[j][1] = siteLoc[1]  # 纬度
            tmp[j][2] = 2018  # 年
            tmp[j][3] = int(((j / 24)+startDate-1) / 30) + 4  # 月
            tmp[j][4] = (int(j / 24) % 30 + startDate - 1) % 30 + 1  # 日
            tmp[j][5] = j % 24  # 小时
            day = (int(j / 24) % 30 + startDate - 1) % 30 + 1
            month = int(((j / 24)+startDate-1) / 30) + 4
            hour = j % 24
            forward = False
            flag = True
            timeSet = df.time.values

            while not (genTimeString(2018, month, day, hour) in timeSet):
                if (forward):
                    hour += 1
                    if (hour == 24):
                        day += 1
                        hour = 0
                        if (day > monthLength[month - 1]):
                            day = 1
                            month += 1
                else:
                    hour -= 1
                    if (hour == -1):
                        day -= 1
                        hour = 23
                        if (day == 0):
                            month -= 1
                            day = monthLength[month - 1]
                if (month == 3):
                    day = (int(j / 24) % 30 + startDate - 1) % 30 + 1
                    month = int(((j / 24)+startDate-1) / 30) + 4
                    hour = j % 24
                    forward = True

            datLine = df[df.time == genTimeString(2018, month, day, hour)]
            if (len(datLine) > 1):
                datLine = datLine.drop_duplicates()
            # print(datLine)

            tmp[j][6] = datLine['PM25_Concentration']
            tmp[j][7] = datLine['PM10_Concentration']
            tmp[j][8] = datLine['NO2_Concentration']
            tmp[j][9] = datLine['CO_Concentration']
            tmp[j][10] = datLine['O3_Concentration']
            tmp[j][11] = datLine['SO2_Concentration']
        newdf = pd.DataFrame(tmp, columns=columnList_aq)
        if (save):
            olddf = pd.read_csv('./processedData/london/' + name + '.csv', header=0, index_col=0)
            fulldf = pd.concat([olddf, newdf], ignore_index=True)
            fulldf.to_csv('./processedData/london/' + name + '.csv')
        df_aq_station.append(newdf)
    # print(df_aq_station[0])

    return df_aq_station


def readProcessedData_bj():
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

    df_aq_station = []
    df_me_station = []
    for i in range(0, len(locList_aq)):
        df = pd.read_csv(filepath_or_buffer='./processedData/beijing/' + locList_aq[i] + '.csv', header=0, index_col=0)
        df_aq_station.append(df)

    for i in range(0, len(locList_me)):
        df = pd.read_csv(filepath_or_buffer='./processedData/beijing/' + locList_me[i] + '.csv', header=0, index_col=0)
        df_me_station.append(df)

    return df_aq_station, df_me_station


def readProcessedData_ld():
    locList_aq = []
    with open('London_AirQuality_Stations.csv', 'r') as locFile:
        line = locFile.readline()
        line = locFile.readline()
        while (line):
            lis = line.split(',')
            if (lis[2] == "TRUE"):
                locList_aq.append(lis[0])
            line = locFile.readline()

    df_aq_station = []
    for i in range(0, len(locList_aq)):
        df = pd.read_csv(filepath_or_buffer='./processedData/london/' + locList_aq[i] + '.csv', header=0, index_col=0)
        df_aq_station.append(df)

    return df_aq_station


startDay = int(input("Enter start day(2018-04-dd):"))
endDay = int(input("Enter end day(2018-04-dd):"))

res_bj = genFrames_bj(startDay, endDay, True)
'''
res_bj = readProcessedData_bj()
res_aq = res_bj[0]
res_me = res_bj[1]
print(res_aq[0])
print(res_me[0])
'''
#genFrames_ld(startDay, endDay, True)
