import pandas
from statistics import mode
from datetime import datetime, date

now = datetime.now()

current_time = now.strftime("%H")
print("Current Time =", current_time)

today = date.today()
print("Today's date:", today)

f = open("/home/ajietb/camera-detection/clockDB.txt", "r")
frameC = f.read().split(" ")
print(frameC)

if not str(today)==frameC[0]:
    if int(current_time) >= 7:
        counter = 0
    elif int(current_time) < 7:
        counter = frameC[1]
else:
    counter = frameC[1]
if frameC[2]=="True":
    clock_reset = True
else:
    clock_reset = False
print("ww", clock_reset)

print(counter)
f.close()

datas = [[2,2,1],
        [2,2,2],
        [2,2],
        [1,1],
        [21,22],
        [1],
        [2]
]

json = {"model":-1,
        "color":-1,
        "pattern":-1,
        "counter":-1
        }

katashiki = {"model":{
                    "d79l":8,
                    "d26":6,
                    "d40l":1,
                    "d40d":4,
                    "d17":5,
                    "d12":2,
                    "t79l":9,
                    "t26":7,
                    },
            "color":{"1g3":1,
                    "4t3":12,
                    "r40":-1,
                    "r71":11,
                    "s28":19,
                    "w09":9,
                    "x12":6,
                    "g64":21,
                    "r75":-1,
                    "p20":20,
                    "1ee7":4,
                    "x09":16,                    
                    },
            "pattern":{
                    "RR-RR-RR":6,
                    "RR-RR":3,
                    "FR-FR":2,
                    "FR-RR":1,
                    "FR":4,
                    "RR":5,
                    "NONE":-1
                    }
            }

pd = pandas.read_csv("/home/ajietb/camera-detection/master_key.csv")

for data in datas:
    num_data = len(data)
    
    if num_data == 3:
        most = int(mode(data))
        pattern = "RR-RR-RR"
    elif num_data == 2:
        if data[0] == data[1]:
            if data[0]%2==0:
                pattern = "RR-RR"
            else:
                pattern = "FR-FR"
        else:
            pattern = "FR-RR"
    elif num_data == 1:
        if data[0]%2==0:
            pattern = "RR"
        else:
            pattern = "FR"
    else:
        pattern = "NONE"
    
    model = str(pd.loc[pd["id"]==most]["model"].values[0])
    color = str(pd.loc[pd["id"]==most]["color"].values[0])

    model = katashiki["model"][str(model)]
    color = katashiki["color"][str(color)]
    pattern = katashiki["pattern"][pattern]

    json["model"] = int(model)
    json["color"] = int(color)
    json["pattern"] = int(pattern)
    json["counter"] = -1

    print(json)