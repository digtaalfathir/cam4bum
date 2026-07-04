import argparse
import time
from pathlib import Path
import threading
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from statistics import mode
import pandas as pds

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from datetime import datetime, date, timedelta
import requests
import os

def detect(save_img=False):
    # Load Data
    analytic_path = f"/opt/camera-sc2/analytic/analytic_ao"
    clockdb=f"/opt/camera-sc2/clockDB-ao.txt"
    pd = pds.read_csv("/opt/camera-sc2/master_key-ao-new.csv")
    uri= "http://192.168.10.49:51003/api/v1/scanner-camera/move-ao"

    #variable
    center_point = 0
    clockdb_status = False
    length_Detect = 0
    croped = True
    detected = {}
    detect = []
    results = None
    send = False
    take_picture = True
    most = 1
    callbk = -1
    bemper_model = -1
    bemper_color = -1
    bemper_pattern = -1

    #Time
    date_now = datetime.now()
    today = date.today()
    timer = time.time()

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
                        "FR-R1-R2-R3-R4":13,
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

    #open clockDB
    f = open(clockdb, "r")
    clockdb_read = f.read().strip().split(" ")
    clockdb_counter = int(clockdb_read[1])
    print("Counter Today : ", clockdb_counter)   
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

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

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

        #read ClockDB
        f = open(clockdb, "r")
        clockdb_date = today
        clockdb_hour = date_now.hour
        clockdb_read = f.read().strip().split(" ")

        if not str(today) == clockdb_read[0]:
            clockdb_status = True
            clockdb_date = today
            f.close()

            w = open(clockdb, "w")
            w.write(f"{clockdb_date} {clockdb_counter} {clockdb_status}")
            w.close()
        else:
            clockdb_status = False
            f.close()
        
        f = open(clockdb, "r")
        clockdb_read = f.read().strip().split(" ")
        if clockdb_hour >= 7 and str(clockdb_read[2]) == "True":
            print("Counter Reset")
            clockdb_counter = 0
            clockdb_status = False
            f.close()

            w = open(clockdb, "w")
            w.write(f"{clockdb_date} {clockdb_counter} {clockdb_status}")
            w.close()
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

            h2 = int(im0.shape[0]/2)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                timer = time.time()
                send = True
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):

                    #whsk
                    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    center_point = int((c1[1] +  c2[1])/2)
                    
                    results = names[int(cls)]

                    if results not in detected:
                        detected[results] = 1
                    else:
                        detected[results] += 1
                        
                    detect = sorted(detected.keys())
                    print("Detected : ", detect)

                    if length_Detect != len(detect):
                        length_Detect = len(detect)
                        take_picture = True


                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                
                    if os.path.isdir(f"{analytic_path}/{str(today)}") == False:
                        os.mkdir(f"{analytic_path}/{str(today)}")
                        for i in range(0, 57):
                            os.mkdir(f"{analytic_path}/{str(today)}/{str(i)}")

                    if take_picture:
                        try:
                            cropped_image = im0[c1[1]-10:c2[1]+10, c1[0]-10:c2[0]+10]
                            cv2.imwrite(f"{analytic_path}/{str(today)}/{str(detect[-1])}/{str(datetime.now().strftime('%X'))}_{detect[-1]}.png", cropped_image)
                            take_picture = False
                        except cv2.error as e:
                            print(f"eror cv2.imwrite :", e)
            else:
                if time.time()-timer>=1:
                    if send:

                        clockdb_counter += 1
                        w = open(clockdb, "w")
                        w.write(f"{clockdb_date} {clockdb_counter} {clockdb_status}")
                        w.close()

                        num_data = len(detect)
                        if num_data>0:
                            global detected_object
                            most=int(mode(detect))
                            detected_objects=detect
                            pattern_result = determine_pattern(detected_objects)
                            if pattern_result not in katashiki["type"]:
                                bemper_pattern = "NONE"
                            else:
                                bemper_pattern=str(pattern_result)
                        else:
                            bemper_pattern="NONE"
                        
                        most = int(most)
                        bemper_model = str(pd.loc[pd["id"]==most]["model"].values[0])
                        bemper_color = str(pd.loc[pd["id"]==most]["color"].values[0])

                        bemper_model = katashiki["model"][bemper_model]
                        bemper_color = katashiki["color"][bemper_color]
                        bemper_pattern = katashiki["type"][bemper_pattern]

                        data["model"] = bemper_model
                        data["color"] = bemper_color
                        data["type"] = bemper_pattern
                        data["counter"] = clockdb_counter
                        data["shiftDate"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        if detected:
                            headers = {'content-type': 'application/json', 'Authorization':'Bearer REDACTED_JWT_TOKEN'}
                            try:
                                r = requests.post(url=uri, json=data, headers=headers)
                                callbk = r.json()
                                print("[DATA] ",data)
                                print(f"[STATUS] {callbk}")
                                yolo_data = 'yolo_data_ao.csv'
                                ydf = pds.DataFrame(data, index=[0])
                                ydf.to_csv(yolo_data, mode='a', header=False, index=False)
                            except requests.exceptions.ConnectionError as e:
                                print (e)
                        else:
                            print(" ➤➤➤ [SEND] EMPTY LIST")
                            counter -= 1
                        send = False
                    detected = {}
                    detect = []
                    take_picture = True
                    data = {"model":-1,
                            "color":-1,
                            "type":-1,
                            "counter":-1,
                            "shiftDate":-1
                            }
                    
            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
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
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')

def determine_pattern(nums):
    def pattern(num):
        if (num<=36) and (num>=1):
            return "FR" if num % 2 == 1 else "RR"
        elif num == 0:
            return "jig"
        elif (num - 36) % 5 == 1:
            return "FR"
        elif (num - 36) % 5 == 2:
            return "R1"
        elif (num - 36) % 5 == 3:
            return "R2"
        elif (num - 36) % 5 == 4:
            return "R3"
        else:
            return "R4"

    nums = [int(num) for num in nums]

    if len(nums) == 1:
        return pattern(nums[0])
    elif len(nums) == 2:
        pat = '-'.join([pattern(num) for num in nums])
    elif len(nums) >= 3:
        pat = '-'.join([pattern(num) for num in nums[:2]]) + '-' + '-'.join([pattern(num) for num in nums[2:]])
    else:
        return "NONE"
    return pat

def updateConnectivity():
    while True:
        os.system("echo $(date +%s) > /opt/camera-sc2/updateConnAO")
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
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    upd_conn_thread = threading.Thread(target=updateConnectivity, name='update connectivity', daemon=True)
    upd_conn_thread.start()

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
