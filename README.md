# cam4bum — Sistem Deteksi & Penghitung Bumper (AO ↔ QI)

Sistem visi komputer berbasis **YOLOv7** untuk mendeteksi, mengklasifikasi, dan
menghitung bumper mobil yang melewati conveyor di dua pos berurutan pada satu line
produksi. Setiap objek dideteksi dua kali — sekali di **AO** dan sekali di **QI** —
sehingga backend dapat merekonsiliasi urutan antrian.

| Pos | Arti | Kamera (RTSP) | Bobot Model | Yang dideteksi |
|-----|------|---------------|-------------|----------------|
| **AO** | *After Oven* — baru keluar proses painting | `172.19.95.34` | `weights/model_SC/AO/best.pt` | model **+ warna** |
| **QI** | *Quality Inspection* | `172.19.95.21` | `weights/best_qi.pt` | model **saja** |

> Karena AO berada tepat setelah painting, warna cat dapat dibaca. QI hanya
> memverifikasi model/tipe, sehingga field `color` selalu `-1`.

---

## Arsitektur

Sistem berjalan sebagai **4 service systemd** — 2 proses deteksi + 2 watchdog:

```
                 ┌───────────────────────── POS AO ─────────────────────────┐
 Kamera AO  ───► │ camera_ao.service → services/after-oven.sh → src/detect-ao.py │
 172.19.95.34    │        │ heartbeat 5s → data/state/updateConnAO               │
                 │        └──────────────► POST /api/v1/scanner-camera/move-ao   │
                 │ checkConnAO (watchdog): heartbeat basi >10s → ping → restart  │
                 └───────────────────────────────────────────────────────────┘
                                            │
                                    (objek yang sama
                                     bergerak ke pos QI)
                                            ▼
                 ┌───────────────────────── POS QI ─────────────────────────┐
 Kamera QI  ───► │ camera_qi.service → services/qi-gate.sh → src/detect-qi.py │
 172.19.95.21    │        │ heartbeat 5s → data/state/updateConnQI               │
                 │        └──────────────► POST /api/v1/scanner-camera/move-qi   │
                 │ checkConnQI (watchdog): heartbeat basi >10s → ping → restart  │
                 │ Stuck.sh   (watchdog): checkStuckQI tak naik 60s → restart    │
                 └───────────────────────────────────────────────────────────┘
                                            │
                                            ▼
                          Backend API  http://192.168.10.49:51003
                          (merekonsiliasi antrian AO ↔ QI)
```

**Pencocokan objek AO↔QI dilakukan di backend, bukan di repo ini.** Kedua pos bekerja
independen dan hanya melaporkan "objek X lewat pada counter N". Dua kamera dipasang
agar backend bisa cross-check urutan antrian dan mengoreksi salah-baca di salah satu pos.

---

## Cara Kerja Deteksi (`src/detect-ao.py` / `src/detect-qi.py`)

Konsepnya seperti **gerbang penghitung / tripwire virtual**:

1. **Baca RTSP stream** → tiap frame di-inference YOLOv7 → NMS → filter kelas tertentu.
2. **Garis scanner virtual** di tengah frame (`h2 = tinggi/2`). Titik-tengah vertikal
   tiap objek (`center_point`) dicek terhadap zona:
   - **AO**: `(h2-150) < center_point < (h2-10)` — zona di **atas** garis tengah
   - **QI**: `(h2+10) < center_point < (h2+150)` — zona di **bawah** garis tengah
   
   (posisi berbeda karena arah gerak / orientasi kamera tiap pos berbeda)
3. **Voting kelas**: selama objek berada di zona, kelasnya di-vote ke dict `centers`
   & `centers2` (kelas yang paling sering terlihat = kandidat terkuat).
4. **Deteksi falling-edge**: saat objek **keluar** zona, kelas mayoritas diambil dan
   dimasukkan ke `packet`. Satu bumper bisa terdiri beberapa bagian (FR=front,
   RR=rear, R1/R2), jadi `packet` bisa berisi >1 elemen.
5. **Frame kosong >4–5 detik** → objek dianggap sudah lewat sepenuhnya → **kirim data**:
   - `determine_pattern()` menerjemahkan urutan bagian menjadi tipe kombinasi
     (mis. `FR-RR`, `R1-R2`).
   - Lookup `config/master_key-*.csv` untuk mendapat model (dan warna, khusus AO).
   - `POST` JSON `{model, color, type, counter, shiftDate}` ke endpoint pos terkait.
   - `counter++`, disimpan ke `data/state/clockDB-*.txt`, di-append ke `data/logs/*.csv`.

**Reset counter harian**: counter otomatis reset ke `0` setiap hari pada jam ≥ 07:00
(awal shift). Nilai counter dipersist ke `clockDB-*.txt` agar bertahan jika program crash.

---

## Struktur Direktori

```
cam4bum/
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/                        # Kode runtime (deteksi)
│   ├── detect-ao.py            #   pipeline pos AO
│   ├── detect-qi.py            #   pipeline pos QI
│   ├── models/                 #   core YOLOv7 (di-import) — JANGAN dipindah
│   └── utils/                  #   core YOLOv7 (di-import) — JANGAN dipindah
│
├── services/                   # Script yang dipanggil systemd
│   ├── after-oven.sh           #   ExecStart camera_ao.service
│   ├── qi-gate.sh              #   ExecStart camera_qi.service
│   ├── checkConnAO             #   watchdog konektivitas AO
│   ├── checkConnQI             #   watchdog konektivitas QI
│   └── Stuck.sh                #   watchdog "stuck" QI
│
├── config/                     # Konfigurasi / master key
│   ├── master_key-ao-new.csv   #   id → model + color (49 baris)
│   └── master_key-qi-new.csv   #   id → model (11 baris)
│
├── weights/                    # Bobot model (di-ignore git; kelola terpisah)
│   ├── model_SC/AO/best.pt     #   bobot AKTIF pos AO
│   ├── model_SC/QI/best.pt     #   bobot alternatif QI
│   ├── best_qi.pt              #   bobot AKTIF pos QI
│   ├── best_ao.pt              #   bobot alternatif AO (tidak dirujuk service)
│   └── traced_model.pt         #   cache TorchScript (auto-regenerate)
│
├── data/                       # Data runtime (mayoritas di-ignore git)
│   ├── state/                  #   state kecil, WAJIB ada saat deploy:
│   │   ├── clockDB-{ao,qi}.txt #     "tanggal counter status_reset"
│   │   ├── updateConn{AO,QI}   #     heartbeat epoch (ditulis tiap 5 detik)
│   │   └── checkStuckQI*.txt   #     counter aktivitas untuk Stuck.sh
│   ├── logs/                   #   log & CSV output (auto-generate)
│   │   ├── yolo_data_{ao,qi}.csv, bumper_detected.csv
│   │   └── connLog{AO,QI}, conection-log.txt
│   └── analytic/               #   snapshot deteksi per hari (analytic_ao/, analytic_qi/)
│
└── archive/                    # Semua file lama / tak dipakai (referensi saja)
    ├── duplicates/             #   copy & versi lama (*_copy, *(copy), *-old, whsky)
    ├── training/               #   skrip training YOLOv7 (train, export, dsb)
    ├── experiments/            #   socket, deep-sort, infinite_loop, konversi CSV
    ├── virtualcam/             #   virtual camera & broadcast (fitur nonaktif)
    ├── monitoring/             #   New Relic (.ini) — nonaktif di kode
    ├── testing/                #   skrip uji manual
    ├── reference/              #   dokumen (Note.txt, xlsx, paper/, figure/, LICENSE)
    └── yolov7-scaffold/        #   scaffolding YOLOv7 asli (runs, tools, cfg, dsb)
```

### Path yang portable (`BASE_DIR`)

Semua path absolut lama (`/opt/camera-sc2/...`) telah diganti dengan path yang
**dihitung dari lokasi file**:

- Python: `BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`
  (folder induk dari `src/`).
- Shell: `BASE_DIR="$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")"`.

Artinya kode berjalan sama baik dari **repo dev** maupun setelah di-deploy ke
`/opt/camera-sc2/` — tanpa perlu edit path lagi.

---

## Deployment

Deployment produksi berada di **`/opt/camera-sc2/`** (salinan dari repo ini).

```bash
# 1) Sinkronkan repo ke lokasi deploy
sudo rsync -av --delete /home/baymax/Program/Git/Github/cam4bum/ /opt/camera-sc2/

# 2) Pastikan bobot model tersedia (tidak ikut git):
#    /opt/camera-sc2/weights/model_SC/AO/best.pt
#    /opt/camera-sc2/weights/best_qi.pt

# 3) Buat file .env dari template lalu isi nilai asli (WAJIB — tidak ikut git):
cp /opt/camera-sc2/.env.example /opt/camera-sc2/.env
#    lalu edit: API_BEARER_TOKEN, RTSP_AO_URL, RTSP_QI_URL

# 4) Reload & restart service
systemctl daemon-reload
systemctl restart camera_ao.service camera_qi.service
```

### 🔐 Konfigurasi secret (`.env`)

Token API dan kredensial RTSP **tidak lagi hardcoded** — dibaca dari file `.env` di
root deployment (di-ignore git). Tanpa `.env`, service tidak akan terhubung ke kamera.

| Variabel | Keterangan |
|----------|------------|
| `API_BEARER_TOKEN` | JWT Bearer untuk backend `scanner-camera` |
| `RTSP_AO_URL` | URL RTSP kamera AO (`rtsp://user:pass@172.19.95.34`) |
| `RTSP_QI_URL` | URL RTSP kamera QI (`rtsp://user:pass@172.19.95.21`) |

`.env` dimuat oleh service script (`services/*.sh` via `set -a; . .env`) **dan** oleh
`detect-*.py` (fungsi `_load_dotenv`), sehingga bisa dijalankan lewat systemd maupun
manual. Template ada di `.env.example` — salin ke `.env` dan isi nilai asli.

### ⚠️ MIGRASI PENTING setelah restrukturisasi ini

Struktur folder berubah, sehingga **unit systemd harus di-update** (file unit ada di
`/etc/systemd/system/` atau `~/.config/systemd/user/`, **tidak** ada di repo ini).
Update `ExecStart` menjadi path baru:

| Service | ExecStart LAMA | ExecStart BARU |
|---------|----------------|----------------|
| `camera_ao.service` | `/opt/camera-sc2/after-oven.sh` | `/opt/camera-sc2/services/after-oven.sh` |
| `camera_qi.service` | `/opt/camera-sc2/qi-gate.sh` | `/opt/camera-sc2/services/qi-gate.sh` |
| watchdog AO | `/opt/camera-sc2/checkConnAO` | `/opt/camera-sc2/services/checkConnAO` |
| watchdog QI | `/opt/camera-sc2/checkConnQI` | `/opt/camera-sc2/services/checkConnQI` |
| watchdog stuck QI | `/opt/camera-sc2/Stuck.sh` | `/opt/camera-sc2/services/Stuck.sh` |

Setelah mengubah unit: `systemctl daemon-reload` lalu restart service terkait.

### Dependensi

```bash
pip install -r requirements.txt   # PyTorch, OpenCV, pandas, requests, pyvirtualcam, dll
```
Butuh GPU CUDA (`--device 0`), `xvfb` (headless display), dan `ffmpeg`/RTSP untuk kamera.

---

## Mekanisme Anti-Macet (berlapis)

Kamera pabrik sering putus, sehingga ada **4 lapis** proteksi:

1. **Heartbeat** — thread di dalam Python menulis epoch ke `data/state/updateConn{AO,QI}`
   tiap 5 detik.
2. **Watchdog konektivitas** (`checkConnAO`/`checkConnQI`) — bila heartbeat basi > 10 detik,
   ping kamera lalu `systemctl restart` service.
3. **Frame-freeze detector** (dalam Python) — bila frame identik persis selama 20 detik,
   proses keluar (`StopIteration`/`sys.exit`) → systemd auto-restart.
4. **Stuck detector** (`Stuck.sh`, khusus QI) — `detect-qi.py` menaikkan
   `checkStuckQI.txt` tiap aktivitas; bila tak bertambah dalam 1 interval (60 dtk) → restart.
   Menangani kasus "proses hidup tapi hang di tengah pipeline".

QI juga punya **auto-reconnect kamera**: bila stream RTSP gagal dibuka, ia mem-ping IP
kamera dalam loop sampai online lagi baru restart.

---

## Konfigurasi Penting

| Parameter | AO | QI |
|-----------|----|----|
| Confidence threshold | `0.95` | `0.91` |
| Kelas yang difilter | `0 3 4..14 17..36` | `0 1..9` |
| Master key | `config/master_key-ao-new.csv` | `config/master_key-qi-new.csv` |
| Endpoint API | `.../move-ao` | `.../move-qi` |
| Field `color` | model + warna | selalu `-1` |
| Voting minimal | `len(centers)==1` | `len(centers)==1 && count>=20` (lebih ketat) |

---

## Catatan & Isu yang Diketahui

- **Inkonsistensi bobot AO/QI**: AO memakai `weights/model_SC/AO/best.pt`, sedangkan
  QI memakai `weights/best_qi.pt` (bukan `model_SC/QI/best.pt`). `weights/best_ao.pt`
  di root **tidak** dirujuk service manapun. Standarisasi disarankan namun belum dilakukan
  agar perilaku tetap identik dengan produksi.
- **State AO tampak lama**: saat restrukturisasi, `updateConnAO` & `clockDB-ao.txt`
  masih tertanggal **Des 2024**, sedangkan QI aktual hari ini — indikasi `camera_ao.service`
  kemungkinan sedang mati/nonaktif. Cek `systemctl status camera_ao.service` bila AO
  seharusnya jalan.
- **Perekaman video nonaktif**: kode memakai `VideoWriter` ke `analytic_*/video/` tapi
  folder `video/` sengaja tidak dibuat, sehingga perekaman menjadi no-op (mencegah disk penuh).
- **New Relic** dinonaktifkan (di-comment di kode; file `.ini` dipindah ke `archive/monitoring/`).

---

## Git

Repo diinisialisasi dengan `.gitignore` yang mengecualikan bobot model (`*.pt`),
media (`*.avi`), artefak training (`runs/`, `wandb/`), serta log & output runtime
(`data/logs/`, `data/analytic/`). Bobot model dikelola terpisah (bukan via git).
