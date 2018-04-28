import requests

def fetch(city_str,day):
    month_str = '04'
    if (day > 30):
        month_str = '05'
        day = day - 30

    day_str = str(day)
    if (day < 10):
        day_str = '0'+day_str

    startTime = '2018-'+month_str + '-' + day_str + '-00'
    url = baseUrl + '/' + city_str + '/' + startTime + '/2k0d1d8'
    respones = requests.get(url)

    filename = './data/forecast/' + city_str + '_' + startTime + '_forecast.csv'

    with open(filename, 'w') as f:
        f.write(respones.text)
        f.close()

startDay = int(input("Enter start day(2018-04-dd):"))

baseUrl = 'http://kdd.caiyunapp.com/competition/forecast'
fetch('bj',startDay)
fetch('ld',startDay)