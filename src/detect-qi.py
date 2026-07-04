import os
# import newrelic.agent

#os.environ['NEW_RELIC_CONFIG_FILE'] = '/opt/camera-sc2/newrelicqi.ini'
#newrelic.agent.initialize('/opt/camera-sc2/newrelicqi.ini')
#application = newrelic.agent.register_application(timeout=10)

import argparse
import time
from pathlib import Path
import sys
import requests
import pandas as pds
from statistics import mode
from datetime import datetime, date, timedelta
#import mysql.connector
#import serverssoc
import threading
import cv2
import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import pyvirtualcam
import logging
import json

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "level": record.levelname,
            "message": record.getMessage(),
            "service": "camera_qi",
            "plant": "sugity_karawang",
        }

        # ambil semua extra fields
        for key, value in record.__dict__.items():
            if key not in (
                "name","msg","args","levelname","levelno","pathname","filename",
                "module","exc_info","exc_text","stack_info","lineno","funcName",
                "created","msecs","relativeCreated","thread","threadName",
                "processName","process","message"
            ):
                log_record[key] = value

        return json.dumps(log_record)

logger = logging.getLogger()
handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

# Root deployment = folder induk dari src/ (parent of this file's dir).
# Membuat semua path portable: jalan di repo dev maupun di /opt/camera-sc2
# tanpa hardcode absolut. Struktur: BASE_DIR/{src,config,weights,data}/...
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_dotenv(path):
    """Muat variabel dari file .env (KEY=VALUE) ke environment bila belum diset.
    Tidak menimpa env yang sudah ada (mis. yang di-export service script)."""
    try:
        with open(path) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    os.environ.setdefault(_k.strip(), _v.strip())
    except FileNotFoundError:
        pass


_load_dotenv(os.path.join(BASE_DIR, ".env"))
# Token Bearer API backend — dari environment / .env (tidak di-commit).
API_BEARER_TOKEN = os.environ.get("API_BEARER_TOKEN", "")

"""
conn=mysql.connector.connect(
    user='root'
    password=''
    host='127.0.0.1'
    database='camera-SC2')
cursor=conn.cursor()
"""

frame_vir = np.ndarray([])
shape_vir = ()

# @newrelic.agent.function_trace()
def detect(save_img=False):
    """
    Deklarasi Variable terkait.
    """
    logger.info("Camera service started", extra={"event":"service_start"})
    # print("[DEBUG Start]")

    global frame_vir
    global shape_vir
    check_stuck = 0
    shape_vir = []
    previous_image = np.ndarray([])
    previous_image_time = time.time()
    packet = []
    centers = {}
    centers2 = {}
    previous_state = False
    clock_state = False
    send = False
    cropped = True
    reset_camera = True
    previous_date = date.today()
    clock_str = "False"
    timer = time.time()
    uri= "http://192.168.10.49:51003/api/v1/scanner-camera/move-qi"
    #sql="""INSERT INTO detect (model)"""
    pd = pds.read_csv(os.path.join(BASE_DIR, "config", "master_key-qi-new.csv")) #master key berisikan kunci untuk setiap model bemper (jenis, warna, dll)
    analytic_path = os.path.join(BASE_DIR, "data", "analytic", "analytic_qi")
    clockdb = os.path.join(BASE_DIR, "data", "state", "clockDB-qi.txt")
    checkStuckLog = os.path.join(BASE_DIR, "data", "state", "checkStuckQI.txt")
    checkStuckLogbckup = os.path.join(BASE_DIR, "data", "state", "checkStuckQILast.txt")
    fps, w, h = 15, 640, 360
    video_date = date.today()
    video_now = datetime.now()
    video_time = video_now.strftime("%H-%M-%S")
    video_writer = cv2.VideoWriter(f"{analytic_path}/video/VID_{video_date}_{video_time}.avi", cv2.VideoWriter_fourcc('M','J','P','G'), fps, (w, h)) #Inisialisasi vid_writer untuk menyimpan model terdeteksi video
    
    check_stuck += 1
    z = open(checkStuckLog, "w")
    z.write(str(check_stuck))
    z.close()
    # print("\n[DEBUG1]", video_date, video_time)
    z = open(checkStuckLogbckup, "w")
    z.write(str(check_stuck))
    z.close()

    most = 1
    callbk = -1
    bemper_model = -1
    bemper_color = -1
    bemper_pattern = -1

    #data json untuk dikirim 
    data = {"model":-1,
            "color":-1,
            "type":-1,
            "counter":-1,
            "shiftDate":-1
            }

    #Dictionary yang berisi list model, color, dan pattern
    katashiki = {"model":{
                        "jig":0,
                        "avanza":1,
                        "calya":2,
                        "veloz":3,
                        "yaris":4,
                        "d03b":5,
                        },
                "color":{
                        "jig":0,
                        "1g3":1,
                        "3q3":2,
                        "g64":3,
                        "s28":4,
                        "w09":5,
                        "x12":6,
                        "4t3":7,
                        "089":8,
                        "040":9,
                        "p20":10,
                        "6w2":11,
                        "218":12,
                        "b88":13,
                        "r77":14,

                        },
                "type":{
                        "RR-RR-RR":6,
                        "FR-R1-R2":7,
                        "FR-R2-R1":7,
                        "R2-R1-FR":7,
                        "R1-R2":8,
                        "R2-R1":8,
                        "FR-R1":9,
                        "R1-FR":9,
                        "FR-R2":10,
                        "R2-FR":10,
                        "R1":11,
                        "R2":12,
                        "RR-RR":3,
                        "FR-FR":2,
                        "FR-RR":1,
                        "RR-FR":1,
                        "FR":4,
                        "RR":5,
                        "NONE":-1,
                        "jig":0
                        }
                }
    
    now = datetime.now()
    current_time = now.strftime("%H")
    today = date.today()

    """
    Terdapat file clockDB.txt yang digunakan untuk menyimpan counter saat ini ke file lokal, 
    sehingga ketika terdapat error yang memaksa program break, maka dapat membaca nilai counter terakhir.
    contoh isi file counterDB.txt

    2022-12-13 0 False
         ^     ^   ^
        date count status(apakah counter sudah reset atau belum hari ini)  
    """
    
    f = open(clockdb, "r")
    frameC = f.read().strip().split(" ")

    counter = int(frameC[1])
    check_stuck += 1
    z = open(checkStuckLog, "w")
    z.write(str(check_stuck))
    z.close()

    f.close()

    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    
    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    logger.info("YOLO model loaded successfully")

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True

        try:
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)

            logger.info(
                "Camera stream initialized",
                extra={
                    "event": "camera_stream_start",
                    "source": source
                }
            )

        except Exception as e:

            logger.error(
                "Camera stream open failed",
                extra={
                    "event": "camera_stream_error",
                    "source": source,
                    "error": str(e)
                }
            )

            logger.warning(
                "Camera offline detected",
                extra={
                    "event": "camera_offline",
                    "source": source
                }
            )

            # ambil IP kamera dari RTSP
            import re
            match = re.search(r'@([\d\.]+)', source)
            camera_ip = match.group(1) if match else None

            if camera_ip is None:
                logger.error(
                    "Failed to parse camera IP",
                    extra={"event": "camera_ip_parse_error"}
                )
                time.sleep(60)
            else:

                logger.info(
                    "Waiting for camera to come back online",
                    extra={
                        "event": "camera_wait_ping",
                        "camera_ip": camera_ip
                    }
                )

                while True:

                    response = os.system(f"ping -c 1 -W 1 {camera_ip} > /dev/null 2>&1")

                    if response == 0:
                        logger.info(
                            "Camera reachable again, restarting service",
                            extra={
                                "event": "camera_online_restart",
                                "camera_ip": camera_ip
                            }
                        )

                        os.system("systemctl restart camera_qi.service")
                        time.sleep(5)
                        sys.exit(0)

                    time.sleep(5)

    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        logger.error(
            "Camera stream failed",
            extra={
                "event": "camera_stream_error",
                "source": source
            }
        )


    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        f = open(clockdb, "r") #Baca file clockDB.txt
        
        #ambil date dan time hari ini
        today = date.today()
        now = datetime.now()
        current_time = now.strftime("%H")
        frameC = f.read().strip().split(" ")
        
        if not str(today)==frameC[0]: #Cek apakah sudah berganti tanngal (tanggal saat ini dibandingkan dengan tanggal yang ada dalam clockDB.txt), jika berbeda maka eksekusi
            clock_state = True
            clock_str = "True" #Mengganti status menjadi True, sehingga pada hari ini dapat melakukkan reset counter
            previous_date = today #meng-update tanggal di clockDB.txt dengan tanggal hari ini
            f.close()

            d = open(clockdb, "w")
            d.write(str(previous_date) + " " + str(counter) + " " + str(clock_str))
            d.close()
        else:
            f.close()            

        f = open(clockdb, "r") #Baca file clockDB.txt
        frameC = f.read().strip().split(" ")
        if int(current_time) >= 7 and str(frameC[2]) == "True": #Cek apakah hari ini jam 7 dan status True, jika iya maka eksekusi program
            logger.warning("Daily counter reset executed")
            counter = 0 #Reset counter menjadi 0
            clock_state = False 
            clock_str = "False" #Mengganti status menjadi False, sehingga hanya sekali dalam sehari program dapat melakukan reset counter
            f.close()

            d = open(clockdb, "w")
            d.write(str(previous_date) + " " + str(counter) + " " + str(clock_str))
            d.close()
        else:
            f.close()

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            
            shape_vir = im0.shape
            frame_vir = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
            w = im0.shape[1]
            h2 = int(im0.shape[0]/2)

            
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            if len(det):
                check_stuck += 1
                z = open(checkStuckLog, "w")
                z.write(str(check_stuck))
                z.close()

                video_writer.write(im0)
                timer = time.time()
                # send = True

                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    check_stuck += 1
                    z = open(checkStuckLog, "w")
                    z.write(str(check_stuck))
                    z.close()

                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to 
                # Write results
                datas = []
                state = False
                for *xyxy, conf, cls in reversed(det):
                    check_stuck += 1
                    z = open(checkStuckLog, "w")
                    z.write(str(check_stuck))
                    z.close()

                    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    center_point = int((c1[1] +  c2[1])/2)
                    
                    if (h2+10)<center_point<(h2+150): #Cek apakah titik tengah objek yang terdeteksi berada pada rentang scanner, jika iya maka mengupdate dictionary kelas
                        """
                        jika kelas sudah ada dalam dictionary maka akan dilakukan increement, jika belum maka akan melakukan inisialisasi counter sebesar 1
                        terdapat dua variable dictionary yaitu, centers dan centers2
                        """
                        if f"{names[int(cls)]}" not in centers:
                            centers[f"{names[int(cls)]}"] = 1
                        else:
                            centers[f"{names[int(cls)]}"]+=1
                        
                        if f"{names[int(cls)]}" not in centers2:
                            centers2[f"{names[int(cls)]}"] = 1
                        else:
                            centers2[f"{names[int(cls)]}"]+=1
                        
                        if cropped: #Crop image, dilakukan untu memvalidasi objek yang terdeteksi apakah sudah sesuai 
                            cropped_image = im0[c1[1]-10:c2[1]+10, c1[0]-10:c2[0]+10]
                            cropped = False

                        datas.append(center_point)
                    
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                if datas:
                    state = True
                else:
                    state = False
                
                """
                if not state:
                    if previous_state:
                        ...
                
                digunakan untuk mengecek signal rising, signal rising merupakan signal ketika terdapat objek terdeteksi yang baru saja memasuki bounding box scanner
                """
                
                if not state:
                    if previous_state:
                        check_stuck += 1
                        z = open(checkStuckLog, "w")
                        z.write(str(check_stuck))
                        z.close()

                        today = date.today()
                        now = datetime.now()
                        current_time = now.strftime("%H-%M-%S")
                        try:
                            if 0<=int(now.strftime("%H"))<7:
                                today = date.today() - timedelta(days=1)
                        except:
                            pass                                                                                                               
                        
                        max_value = max(centers, key=centers.get)
                        list_value =list(centers.keys())
                        count_s=int(''.join(map(str, centers.values())))
                        if (len(centers)==1) and (count_s >= 20):
                            packet.append(list(centers.keys())[0])
                        else:
                            key=list(centers.keys())[0]                            
                            if int(key)<41:
                                packet.append(max_value)
                            else:
                                for x in list_value:
                                    packet.append(x)
                        centers = {}
                        
                        if str(today) not in os.listdir(analytic_path):
                            os.mkdir(analytic_path+"/"+str(today))

                        if str(packet[-1]) not in os.listdir(analytic_path+"/"+str(today)):
                            os.mkdir(analytic_path+"/"+str(today)+f"/{packet[-1]}")
                        
                check_stuck += 1
                z = open(checkStuckLog, "w")
                z.write(str(check_stuck))
                z.close()

                previous_state = state
                
            else:
                #Jika dalam frame sudah tidak terdapat objek yang terdeteksi, maka lakukan pengiriman API json
                check_stuck += 1
                if check_stuck % 5000 == 0:
                    logger.warning(
                        "No detection for extended period",
                        extra={
                            "event": "no_detection_warning",
                            "stuck_counter": check_stuck
                        }
                    )
                z = open(checkStuckLog, "w")
                z.write(str(check_stuck))
                z.close()
                if time.time()-timer>4: #Tunggu sekitar 2 detik untuk memastikan frame benar benar kosong
                    if centers2:  
                        check_stuck += 1 
                        z = open(checkStuckLog, "w")
                        z.write(str(check_stuck))
                        z.close()
                        if len(packet)==0 and len(centers2)==1:
                            packet.append(list(centers2.keys())[0])
                            centers = {}

                        now = datetime.now()
                        current_time = now.strftime("%H")
                        today = date.today()

                        num_data=len(packet)
                        if num_data>0:                  
                            global detected_object
                            most=int(mode(packet))
                            if packet[0] != 8:
                                detected_objects=packet[:2]
                            else:
                                detected_objects=packet[:3]
                            pattern_result = determine_pattern(detected_objects)
                            if pattern_result not in katashiki["type"]:
                                bemper_pattern = "NONE"
                            else:
                                bemper_pattern=str(pattern_result)

                        else:
                            bemper_pattern="NONE"
                            
                        most = int(most)
                        bemper_model = str(pd.loc[pd["id"]==most]["model"].values[0])

                        bemper_model = katashiki["model"][str(bemper_model)]
                        bemper_pattern = katashiki["type"][bemper_pattern]

                        data["model"] = int(bemper_model)
                        data["color"] = -1
                        data["type"] = int(bemper_pattern)
                        data["counter"] = counter
                        data["shiftDate"] = now.strftime("%Y-%m-%d %H:%M:%S")
                        headers = {'content-type': 'application/json', 'Authorization': 'Bearer ' + API_BEARER_TOKEN}
                        try:
                            logger.info(
                                "Sending detection to API",
                                extra={
                                    "event": "api_request",
                                    "endpoint": uri,
                                    "payload": data
                                }
                            )
                            r = requests.post(url=uri, json=data, headers=headers)
                            callbk = r.json()
                            logger.info(
                                "API success",
                                extra={
                                    "event": "api_success",
                                    "status_code": r.status_code,
                                    "response": callbk,
                                    "model": bemper_model,
                                    "pattern": bemper_pattern,
                                    "counter": counter
                                }
                            )
                            check_stuck = 0
                            z = open(checkStuckLogbckup, "w")
                            z.write(str(check_stuck))
                            z.close()
                            
                            counter += 1

                            f = open(clockdb, "r")
                            frameC = f.read().strip().split(" ")
                            f.close()

                            d = open(clockdb, "w")
                            d.write(str(frameC[0]) + " " + str(counter) + " " + str(frameC[2]))
                            d.close()

                            yolo_data = os.path.join(BASE_DIR, "data", "logs", "yolo_data_qi.csv")
                            ydf = pds.DataFrame(data, index=[0])
                            ydf.to_csv(yolo_data, mode='a', header=False, index=False)
                        except requests.exceptions.ConnectionError as e:
                            logger.error(
                                "API request failed",
                                extra={
                                    "event": "api_error",
                                    "endpoint": uri,
                                    "payload": data,
                                    "error": str(e)
                                }
                            )                        
                    
                    centers2 = {}
                    packet = []
                    previous_state = False
                    data = {"model":-1,
                            "color":-1,
                            "type":-1,
                            "counter":-1,
                            "shiftDate":-1
                            }

            # Stream results
            if view_img or True:
                if time.time() - previous_image_time>20: #digunakan untuk mengantisipasi terdapat freeze kamera, maka program akan dipaksa break, dan di restart program
                    if np.array_equal(im0, previous_image):
                        cv2.destroyAllWindows()
                        with open(os.path.join(BASE_DIR, "data", "logs", "conection-log.txt"),'a') as file:
                            con_log= now.strftime("%Y-%m-%d %H:%M:%S") + "  CAMERA-QI got FRAME FREEZE, check LAN connection!!"
                            file.write(con_log + '\n')
                        logger.critical(
                            "Frame freeze detected - restarting service",
                            extra={"event": "frame_freeze"}
                        )
                        sys.exit(1)
                    previous_image = im0.copy()
                    previous_image_time = time.time()

                cv2.imshow(str("camera-opened"), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        logger.info(
                            "Video recording started",
                            extra={
                                "event": "video_record_start",
                                "file": f"{analytic_path}/video/VID_{video_date}_{video_time}.avi"
                            }
                        )
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
                    

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''

def determine_pattern(nums):
    def pattern(num):
        if (num<=8) and (num>=0):
            return "FR" if num % 2 == 0 else "RR"
        elif (num == 9):
            return "R1-R2"

    nums = [int(num) for num in nums]

    if len(nums) == 1:
        return pattern(nums[0])
    elif len(nums) == 2:
        pat = '-'.join([pattern(num) for num in nums])
    elif len(nums) == 3:
        pat = '-'.join([pattern(num) for num in nums[:2]]) + '-' + '-'.join([pattern(num) for num in nums[2:]])
    else:
        return "NONE"
    return pat
                        
def updateConnectivity():
    while True:
        os.system("echo $(date +%s) > " + os.path.join(BASE_DIR, "data", "state", "updateConnQI"))
        time.sleep(5)                
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()

    upd_conn_thread = threading.Thread(target=updateConnectivity, name='update connectivity', daemon=True)
    upd_conn_thread.start()

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()

