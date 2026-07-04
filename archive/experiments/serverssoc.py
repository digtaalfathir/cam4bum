# Import the required modules
from IPython.display import clear_output
import socket
import sys
import cv2
import matplotlib.pyplot as plt
import pickle
import numpy as np
import struct ## new
import zlib
from PIL import Image, ImageOps
import pyvirtualcam

HOST=''
PORT=8089

s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
print('Socket created')

s.bind((HOST,PORT))
print('Socket bind complete')
s.listen(10)
print('Socket now listening')

conn,addr=s.accept()

data = b""
payload_size = struct.calcsize(">L")
print("payload_size: {}".format(payload_size))
with pyvirtualcam.Camera(width=640, height=320, fps=20, print_fps=True) as cam:
    print(f'Using virtual camera: {cam.device}')
    while True:
        while len(data) < payload_size:
            data += conn.recv(4096)
            if not data:
                cv2.destroyAllWindows()
                conn,addr=s.accept()
                continue
        # receive image row data form client socket
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]
        while len(data) < msg_size:
            data += conn.recv(4096)
        frame_data = data[:msg_size]
        data = data[msg_size:]
        # unpack image using pickle 
        
        frame=pickle.loads(frame_data, fix_imports=True, encoding="bytes")
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cam.send(frame)
        cam.sleep_until_next_frame()

        cv2.imshow('server',frame)
        cv2.waitKey(1)

 
    """frame = np.zeros((cam.height, cam.width, 3), np.uint8)  # RGB
    while True:
        h, s, v = (cam.frames_sent % 100) / 100, 1.0, 1.0
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        frame[:] = (r * 255, g * 255, b * 255)
        cam.send(frame)
        cam.sleep_until_next_frame()"""