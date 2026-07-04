#!/bin/bash

modprobe -r v4l2loopback
modprobe v4l2loopback devices=2 video_nr=3,4 card_label="Virtual1","Virtual2" exclusive_caps=1 max_buffers=2
echo "🚀🚀🚀 Virtual Camera Opened"
echo "|"
echo "|"

