import os
import numpy as np
import wradlib as wrl
from tqdm import tqdm  # for progress bar
import json

# Set WRADLIB_DATA folder
data_dir = "wradlib_data"
os.environ['WRADLIB_DATA'] = data_dir

# Ensure output directory exists
output_dir = "Data"
os.makedirs(output_dir, exist_ok=True)

# Target padded shape
TARGET_H, TARGET_W = 360, 240

# Helper function to process one file
def process_one_file(file_path):
    data, _ = wrl.io.read_gamic_hdf5(file_path)

    processed_scans = []
    for i in tqdm(range(14), desc=f"Scans in {os.path.basename(file_path)}", leave=False):
        scan_key = f"SCAN{i}"

        if "ZH" not in data[scan_key]:
            raise ValueError(f"ZH not found in {scan_key} of file {file_path}")

        arr = data[scan_key]["ZH"]["data"]
        arr[arr == 96.00197] = 0  # clean

        h, w = arr.shape

        # Pad height
        if h > TARGET_H:
            arr = arr[:TARGET_H, :]
        elif h < TARGET_H:
            pad_bottom = TARGET_H - h
            arr = np.pad(arr, ((0, pad_bottom), (0, 0)), mode='constant', constant_values=0)

        # Pad width
        if w > TARGET_W:
            arr = arr[:, :TARGET_W]
        elif w < TARGET_W:
            pad_right = TARGET_W - w
            arr = np.pad(arr, ((0, 0), (0, pad_right)), mode='constant', constant_values=0)

        processed_scans.append(arr)

    return np.stack(processed_scans)  # shape: (14, 360, 240)

# Recursively find all .h5 files
files = []
for root, dirs, filenames in os.walk(data_dir):
    for fname in filenames:
        if fname.endswith(".h5"):
            files.append(os.path.join(root, fname))

print(f"Found {len(files)} HDF5 files to process.")

all_data = []

# Sort files for reproducibility
files = sorted(files)

for fname in tqdm(files, desc="Processing files"):
    try:
        tensor = process_one_file(fname)  # (14, 360, 240)
        all_data.append(tensor)
    except Exception as e:
        print(f"Error processing {fname}: {e}")

# Final dataset shape: (num_files, 14, 360, 240)
dataset = np.stack(all_data)
print("Final dataset shape:", dataset.shape)

# Save to .npy file
np.save("Data/ZH_radar_dataset.npy", dataset)
print("Saved dataset to Data/ZH_radar_dataset.npy")

# Save the sorted filenames as a .json file (relative to data_dir)
rel_files = [os.path.relpath(f, data_dir) for f in files]
with open("Data/ZH_radar_filenames.json", "w") as f:
    json.dump(rel_files, f)
print("Saved filenames to Data/ZH_radar_filenames.json") 