import os
import time

while True:
    try:
        print("wait 5s")
        time.sleep(5)
        os.system("python3 detect-ao.py --weights best5.pt --source rtsp://172.19.95.33:8089/video_stream --img-size 320 --nosave --conf 0.9")
    except:
        pass
