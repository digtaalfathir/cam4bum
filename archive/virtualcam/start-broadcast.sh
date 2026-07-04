#!/bin/bash

sudo modprobe -r v4l2loopback
sudo modprobe v4l2loopback devices=2 video_nr=3,4 card_label="Camera AO","Camera QI" exclusive_caps=1,0 max_buffers=2
echo "🚀🚀🚀 Virtual Camera Opened"
echo "|"
echo "|"

