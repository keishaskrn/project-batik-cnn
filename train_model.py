"""
train_model.py
Klasifikasi Jenis Batik Berdasarkan Motif Menggunakan CNN
Jalankan file ini untuk melatih model CNN dan menyimpannya ke folder model/
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# KONFIGURASI
# ============================================================
DATASET_DIR   = "dataset"        # Folder dataset (isi sesuai struktur di bawah)
IMG_SIZE      = (128, 128)
BATCH_SIZE    = 32
EPOCHS        = 20
MODEL_PATH    = "model/batik_model.h5"
LABELS_PATH   = "model/class_names.txt"

# ============================================================
# STRUKTUR DATASET YANG DIHARAPKAN:
#
# dataset/
# ├── train/
# │   ├── batik_parang/
# │   ├── batik_kawung/
# │   ├── batik_mega_mendung/
# │   ├── batik_truntum/
# │   └── batik_sekar_jagad/
# └── val/
#     ├── batik_parang/
#     ├── batik_kawung/
#     ├── batik_mega_mendung/
#     ├── batik_truntum/
#     └── batik_sekar_jagad/
#
# Sumber dataset: https://www.kaggle.com/datasets/dionisiusdh/indonesian-batik-motifs
#                 atau cari "batik classification dataset" di Kaggle
# ============================================================

def load_data():
    train_ds = keras.preprocessing.image_dataset_from_directory(
        os.path.join(DATASET_DIR, "train"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True,
        seed=42
    )
    val_ds = keras.preprocessing.image_dataset_from_directory(
        os.path.join(DATASET_DIR, "val"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
        seed=42
    )
    return train_ds, val_ds

def build_model(num_classes):
    """
    Custom CNN dari awal (sesuai tugas).
    Arsitektur: 3 blok Conv+MaxPool → Flatten → Dense → Dropout → Output
    """
    model = keras.Sequential([
        # Input normalisasi pixel ke [0, 1]
        layers.Rescaling(1./255, input_shape=(128, 128, 3)),

        # Augmentasi data (hanya aktif saat training)
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),

        # Blok Konvolusi 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Blok Konvolusi 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Blok Konvolusi 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Blok Konvolusi 4
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Fully Connected
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),

        # Output Layer
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def plot_history(history, save_path="model/training_history.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"[INFO] Grafik training disimpan ke {save_path}")

def main():
    print("=" * 60)
    print("  TRAINING CNN - KLASIFIKASI MOTIF BATIK INDONESIA")
    print("=" * 60)

    # Load data
    print("\n[INFO] Memuat dataset...")
    train_ds, val_ds = load_data()
    class_names = train_ds.class_names
    num_classes  = len(class_names)
    print(f"[INFO] Kelas terdeteksi ({num_classes}): {class_names}")

    # Simpan nama kelas
    os.makedirs("model", exist_ok=True)
    with open(LABELS_PATH, "w") as f:
        for name in class_names:
            f.write(name + "\n")
    print(f"[INFO] Nama kelas disimpan ke {LABELS_PATH}")

    # Optimasi pipeline
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Bangun model
    print("\n[INFO] Membangun model CNN...")
    model = build_model(num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    # Training
    print(f"\n[INFO] Mulai training selama {EPOCHS} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Evaluasi
    print("\n[INFO] Evaluasi model pada data validasi...")
    val_loss, val_acc = model.evaluate(val_ds)
    print(f"\n[HASIL] Val Loss    : {val_loss:.4f}")
    print(f"[HASIL] Val Accuracy: {val_acc*100:.2f}%")

    # Simpan grafik
    plot_history(history)

    print(f"\n[SELESAI] Model berhasil disimpan ke '{MODEL_PATH}'")
    print("Sekarang jalankan: python app.py")

if __name__ == "__main__":
    main()
