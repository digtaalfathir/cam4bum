#!/bin/bash

DISPLAY_NUMBER=:1

LOG_FILE="lox_dummy.log"
CONFIG_FILE="/etc/X11/xorg.conf.d/dummy.conf"

Xorg $DISPLAY_NUMBER -noreset +extension GLX +extension RANDR +extension RENDER \-logfile $LOG_FILE -config $CONFIG_FILE &

sleep 2

Xvfb :1 -screen 0 1024x768x16 &
export DISPLAY=$DISPLAY_NUMBER

/usr/bin/python3 /opt/camera-sc2/detect-ao.py --weights best.pt --conf 0.95 --img-size 640 --device 0 --source rtsp://admin:REDACTED@172.19.95.34 --classes 0 1 2 3 4 7 8 9 10 11 12 13 14 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 --nosave

wait
