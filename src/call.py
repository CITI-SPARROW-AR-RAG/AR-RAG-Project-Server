import requests

API_URL = "http://localhost:8000/query"

question = "What is the purpose of this manual?"
payload = {"question": question}

response = requests.post(API_URL, json=payload)

if response.status_code == 200:
    data = response.json()
    print(data)
else:
    print("Error:", response.json())