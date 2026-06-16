from PIL import Image
import os

dataset_path = "dataset"

bad_files = []

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        path = os.path.join(root, file)

        try:
            img = Image.open(path)
            img.verify()
        except Exception:
            bad_files.append(path)

print("\nFILE BERMASALAH:")
for f in bad_files:
    print(f)

print(f"\nTotal file rusak: {len(bad_files)}")