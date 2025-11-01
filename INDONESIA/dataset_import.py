import kagglehub
import os

# Download latest version
path = kagglehub.dataset_download("ashishsaxena2209/animal-image-datasetdog-cat-and-panda")

print("Path to dataset files:", path)

# The dataset will be downloaded to a cache directory
# Copy or move it to your project
import shutil
destination = './INDONESIA/dataset'
os.makedirs(destination, exist_ok=True)

# Copy files from downloaded path to your project
for item in os.listdir(path):
    s = os.path.join(path, item)
    d = os.path.join(destination, item)
    if os.path.isdir(s):
        shutil.copytree(s, d, dirs_exist_ok=True)
    else:
        shutil.copy2(s, d)

print(f"Dataset copied to {destination}")