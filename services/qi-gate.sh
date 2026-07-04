#!/bin/bash
# Service: camera_qi -- menjalankan deteksi objek kamera QI (Quality Inspection).
# Path dihitung relatif terhadap lokasi script -> portable (repo dev / /opt/camera-sc2).
sleep 5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

# Muat secret (RTSP_QI_URL, API_BEARER_TOKEN, ...) dari .env — tidak di-commit.
if [ -f "$BASE_DIR/.env" ]; then
    set -a
    . "$BASE_DIR/.env"
    set +a
fi

xvfb-run -a -s "-screen 0 1024x768x16" /usr/bin/python3 "$BASE_DIR/src/detect-qi.py" \
    --weights "$BASE_DIR/weights/best_qi.pt" \
    --conf 0.91 --img-size 640 --device 0 \
    --source "$RTSP_QI_URL" \
    --nosave --exist-ok \
    --classes 0 1 2 3 4 5 6 7 8 9
