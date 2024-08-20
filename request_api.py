import requests

url = 'http://0.0.0.0:1115'
data = {
    'instruction': '给我讲个笑话',
    'input': ''
}

response = requests.post(url, json=data)

if response.status_code == 200:
    print('Response:', response.text)
else:
    print('Failed to get response:', response.status_code)