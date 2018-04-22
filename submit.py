import requests

files = {'files': open('sample_submission.csv', 'rb')}
data = {
    "user_id": "solitaryzero",
    "team_token": "c7e4335fcf7fced7dd1fc6a7e2adf50057451d226a7e57b759cb62577c72502a",
    "description": 'i am description',
    "filename": "'i am filename",
}
url = 'https://biendata.com/competition/kdd_2018_submit/'
response = requests.post(url, files=files, data=data)
print(response.text)