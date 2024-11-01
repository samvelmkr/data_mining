import requests
import json

def get_input():
    x = float(input("Enter the value for x: "))
    y = float(input("Enter the value for y: "))
    return x, y

def send_request(x, y):
    url = "http://127.0.0.1:8000/predict/"
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"x": x, "y": y})

    response = requests.post(url, headers=headers, data=data)

    if response.status_code == 200:
        print("Response from server:")
        print(response.json())
    else:
        print("Error:", response.status_code, response.text)

if __name__ == "__main__":
    x, y = get_input()
    send_request(x, y)

