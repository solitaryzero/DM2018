import requests

def fetch(type_str,city_str,day):
    month_str = '04'
    if (day > 30):
        month_str = '05'
        day = day - 30

    day_str = str(day)
    if (day < 10):
        day_str = '0'+day_str

    startTime = '2018-'+month_str + '-' + day_str + '-00'
    endTime = '2018-'+month_str + '-' + day_str + '-23'
    url = baseUrl + type_str + '/' + city_str + '/' + startTime + '/' + endTime + '/2k0d1d8'
    respones = requests.get(url)

    if (city_str.endswith('_grid')):
        filename = './data/' + city_str.replace('_grid','') + '_' + type_str + '_grid_' + startTime + '-' + endTime + '.csv'
    else:
        filename = './data/' + city_str + '_' + type_str + '_' + startTime + '-' + endTime + '.csv'

    with open(filename, 'w') as f:
        f.write(respones.text)
        f.close()

startDay = int(input("Enter start day(2018-04-dd):"))
endDay = int(input("Enter end day(2018-04-dd):"))

baseUrl = 'https://biendata.com/competition/'
for i in range(startDay,endDay+1):
    fetch('airquality','bj',i)
    fetch('airquality','ld',i)
    fetch('meteorology','bj',i)
    fetch('meteorology','ld',i)
    fetch('meteorology','bj_grid',i)
    fetch('meteorology','ld_grid',i)