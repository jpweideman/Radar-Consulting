import os
import numpy as np
import wradlib as wrl
from tqdm import tqdm  # for progress bar
import json

# Set WRADLIB_DATA folder
data_dir = "wradlib_data"
os.environ['WRADLIB_DATA'] = data_dir

# Output root directory for processed data
output_root = "Data"
os.makedirs(output_root, exist_ok=True)

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

# Recursively process and save by directory
for root, dirs, files in os.walk(data_dir):
    # Sort year and month directories numerically if possible
    dirs.sort(key=lambda x: int(x) if x.isdigit() else x)
    files.sort()
    h5_files = [f for f in files if f.endswith(".h5")]
    if not h5_files:
        continue
    # Create corresponding output directory
    rel_dir = os.path.relpath(root, data_dir)
    out_dir = os.path.join(output_root, rel_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_npy = os.path.join(out_dir, "data.npy")
    if os.path.exists(out_npy):
        print(f"Skipping {out_dir} (already processed)")
        continue
    tensors = []
    rel_filenames = []
    for fname in tqdm(sorted(h5_files), desc=f"Processing {rel_dir}"):
        fpath = os.path.join(root, fname)
        try:
            tensor = process_one_file(fpath)
            tensors.append(tensor)
            rel_filenames.append(os.path.relpath(fpath, data_dir))
        except Exception as e:
            print(f"Error processing {fpath}: {e}")
    if tensors:
        np.save(out_npy, np.stack(tensors))
        with open(os.path.join(out_dir, "filenames.json"), "w") as f:
            json.dump(rel_filenames, f)
        print(f"Saved {len(tensors)} tensors to {out_npy}")

print("\nProcessing complete!")