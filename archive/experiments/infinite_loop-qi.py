import os
import time

os.system('xvfb-run -a -s -screen O 1024x768x16')
while True:
    try:
        print("wait 2s")
        time.sleep(2)
        os.system("python3 /opt/camera-sc2/detect-qi.py --weights /home/stechoq/Documents/camera-detection/best.pt --source rtsp://admin:REDACTED@172.19.95.21 --img-size 640 --nosave --conf 0.8")
    except:
        pass
