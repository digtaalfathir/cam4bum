import csv

# Dictionary katashiki
katashiki = {
    "model": {
        0: "jig",
        1: "avanza",
        2: "calya",
        3: "veloz",
        4: "yaris",
        5: "d03b"
    },
    "color": {
        0: "jig",
        1: "1g3",
        2: "3q3",
        3: "g64",
        4: "s28",
        5: "w09",
        6: "x12",
        7: "4t3",
        8: "089",
        9: "040",
        10: "p20",
        11: "6w2",
        12: "218",
        13: "b88",
        14: "r77"
    },
    "type": {
        6: "RR-RR-RR",
        7: "FR-R1-R2",
        8: "R1-R2",
        9: "FR-R1",
        10: "FR-R2",
        11: "R1",
        12: "R2",
        3: "RR-RR",
        2: "FR-FR",
        1: "FR-RR",
        4: "FR",
        5: "RR",
        -1: "NONE",
        0: "jig"
    }
}

# Fungsi untuk mengubah data berdasarkan dictionary katashiki
def modify_data_reverse(row):
    try:
        # Pastikan data di index 0, 1, dan 2 berupa angka sebelum konversi
        row[0] = katashiki["model"].get(int(row[0]), row[0])  # Modifikasi kolom model
        row[1] = katashiki["color"].get(int(row[1]), row[1])  # Modifikasi kolom color
        row[2] = katashiki["type"].get(int(row[2]), row[2])   # Modifikasi kolom type
    except ValueError:
        # Lewati baris yang tidak bisa dikonversi (misalnya header atau baris yang berisi teks)
        pass
    return row

# Fungsi utama untuk membaca, mengubah, dan menyimpan kembali ke CSV
def process_csv_reverse(input_file, output_file):
    with open(input_file, mode='r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        rows = list(csvreader)  # Membaca semua baris

    # Memodifikasi setiap baris data
    modified_rows = [modify_data_reverse(row) for row in rows]

    # Menulis hasil modifikasi ke file baru
    with open(output_file, mode='w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(modified_rows)

# Nama file input dan output
input_file = '/opt/camera-sc2/yolo_data_ao.csv'
output_file = '/opt/camera-sc2/convert_yolo_data_ao.csv'

# Memproses CSV
process_csv_reverse(input_file, output_file)
