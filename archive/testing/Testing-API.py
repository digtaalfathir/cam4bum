import requests
from datetime import datetime, date, timedelta

uri= "https://be-karawang.sugity.stehoq.com/api/v1/scanner-camera/move-ao"

data = {"model":-1,
        "color":-1,
        "type":-1,
        "counter":-1,
        "shiftDate":-1
        }

now = datetime.now()
data["model"] = 1
data["color"] = 2
data["type"] = 3
data["counter"] = 4
data["shiftDate"] = now.strftime("%Y-%m-%d %H:%M:%S")

headers = {'content-type': 'application/json', 'Authorization':'Bearer REDACTED_JWT_TOKEN'}

try:
    r = requests.post(url=uri, json=data, headers=headers)
    callbk = r.json()
    print (callbk)
    print("[DATA] ",data)
except requests.exceptions.ConnectionError as e:
    print (e)


print ("masuk sini")