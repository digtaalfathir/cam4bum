#!/bin/bash
# Watchdog "stuck" QI: mendeteksi program hidup tapi macet di tengah pipeline.
# detect-qi.py meng-increment data/state/checkStuckQI.txt tiap aktivitas.
# Bila nilainya tidak bertambah dalam 1 interval -> restart service camera_qi.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

while true; do
    # File counter aktivitas yang ditulis detect-qi.py
    file="$BASE_DIR/data/state/checkStuckQI.txt"
    # File untuk menyimpan nilai terakhir yang dibaca
    last_value_file="$BASE_DIR/data/state/checkStuckQILast.txt"
    # Nama service yang akan di-restart
    service_name="camera_qi.service"
    echo "Service akan terjeda 5 menit"
    sleep 60
    echo "Service jalan setelah terjeda 5 menit"
    # Periksa apakah file utama ada
    if [[ ! -f "$file" ]]; then
        echo "File $file tidak ditemukan!"
        exit 1
    fi

    # Ambil nilai terakhir dari file
    current_value=$(cat "$file")

    # Cek apakah file untuk menyimpan nilai terakhir ada
    if [[ ! -f "$last_value_file" ]]; then
        # Jika tidak ada, simpan nilai pertama kali
        echo "$current_value" > "$last_value_file"
        exit 0
    fi

    # Baca nilai terakhir yang disimpan
    last_value=$(cat "$last_value_file")

    # Bandingkan nilai terakhir dengan nilai saat ini
    if [[ "$current_value" -le "$last_value" ]]; then
        echo "Nilai tidak bertambah, merestart service $service_name..."
        # Restart service jika nilai tidak bertambah
        sudo systemctl restart "$service_name"
    else
        # Simpan nilai terbaru jika ada perubahan
        echo "$current_value" > "$last_value_file"
        echo "Nilai bertambah, tidak perlu merestart service."
    fi
done
