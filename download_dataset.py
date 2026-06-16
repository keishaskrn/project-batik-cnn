import kagglehub
import shutil
import os

print("Downloading dataset...")

path = kagglehub.dataset_download(
    "dionisiusdh/indonesian-batik-motifs"
)

print("Dataset downloaded at:")
print(path)