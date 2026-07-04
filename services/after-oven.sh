#!/bin/bash
# Service: camera_ao -- menjalankan deteksi objek kamera AO (After Oven).
# Path dihitung relatif terhadap lokasi script -> portable (repo dev / /opt/camera-sc2).
sleep 5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

# Muat secret (RTSP_AO_URL, API_BEARER_TOKEN, ...) dari .env — tidak di-commit.
if [ -f "$BASE_DIR/.env" ]; then
    set -a
    . "$BASE_DIR/.env"
    set +a
fi

xvfb-run -a -s "-screen 0 1024x768x16" /usr/bin/python3 "$BASE_DIR/src/detect-ao.py" \
    --weights "$BASE_DIR/weights/model_SC/AO/best.pt" \
    --conf 0.95 --img-size 640 --device 0 \
    --source "$RTSP_AO_URL" \
    --classes 0 3 4 5 6 7 8 9 10 11 12 13 14 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 \
    --nosave
