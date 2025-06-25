import requests

url = "http://127.0.0.1:8000/predict_genre/"
data = {"file_path": "F:/AIU/2024/Senior/classification/test/50cent.wav"}

response = requests.post(url, json=data)
print(response.json())
