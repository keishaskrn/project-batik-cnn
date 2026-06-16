"""
setup_dataset.py
Helper script untuk membuat struktur folder dataset secara otomatis.
Jalankan SEBELUM train_model.py jika kamu baru mulai.
"""

import os

CLASSES = [
    "batik_parang",
    "batik_kawung",
    "batik_mega_mendung",
    "batik_truntum",
    "batik_sekar_jagad"
]

def create_dataset_structure():
    print("=" * 55)
    print("  SETUP STRUKTUR FOLDER DATASET BATIK")
    print("=" * 55)

    for split in ["train", "val"]:
        for cls in CLASSES:
            path = os.path.join("dataset", split, cls)
            os.makedirs(path, exist_ok=True)
            print(f"[OK] Folder dibuat: {path}")

    print("\n✅ Struktur folder berhasil dibuat!")
    print("\n📋 Langkah selanjutnya:")
    print("   1. Download dataset dari Kaggle:")
    print("      https://www.kaggle.com/datasets/dionisiusdh/indonesian-batik-motifs")
    print("   2. Salin gambar ke folder yang sesuai:")
    print("      - 80% gambar → dataset/train/<nama_kelas>/")
    print("      - 20% gambar → dataset/val/<nama_kelas>/")
    print("   3. Jalankan: python train_model.py")
    print("   4. Jalankan: python app.py")
    print("\n💡 Tips: Minimal 50 gambar per kelas untuk hasil yang baik.")
    print("         Semakin banyak gambar, semakin akurat modelnya.")

if __name__ == "__main__":
    create_dataset_structure()
