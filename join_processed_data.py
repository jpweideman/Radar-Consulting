import os
import numpy as np
import json
from tqdm import tqdm

data_arrays = []
all_filenames = []

# First, collect all relevant directories for progress bar
join_targets = []   
#for root, dirs, files in os.walk('Data'):           # for data in remote directory
for root, dirs, files in os.walk('/tmp/Data'):      # for data in local directory
    dirs.sort(key=lambda x: int(x) if x.isdigit() else x)
    files.sort()
    if 'data.npy' in files and 'filenames.json' in files:
        join_targets.append(root)

# Determine total number of samples and sample shape
total_samples = 0
sample_shape = None
for root in join_targets:
    arr = np.load(os.path.join(root, 'data.npy'), mmap_mode='r')
    if sample_shape is None:
        sample_shape = arr.shape[1:]  # (14, 360, 240)
    total_samples += arr.shape[0]

# Ensure output directory exists
os.makedirs('Data', exist_ok=True)

# Pre-allocate memmap array for output
out_path = 'Data/ZH_radar_dataset.npy'
final_data = np.lib.format.open_memmap(out_path, mode='w+', dtype='float32', shape=(total_samples, *sample_shape))

# Fill the memmap array and join filenames
idx = 0
all_filenames = []
for root in tqdm(join_targets, desc="Joining processed data"):
    arr = np.load(os.path.join(root, 'data.npy'))
    n = arr.shape[0]
    final_data[idx:idx+n] = arr
    idx += n
    with open(os.path.join(root, 'filenames.json')) as f:
        names = json.load(f)
        all_filenames.extend(names)

# Save filenames
with open('Data/ZH_radar_filenames.json', 'w') as f:
    json.dump(all_filenames, f)
print(f"Saved concatenated data to {out_path}, shape: {final_data.shape}")
print(f"Saved concatenated filenames to Data/ZH_radar_filenames.json, count: {len(all_filenames)}") 