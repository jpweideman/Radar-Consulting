import os
import numpy as np
import json
from tqdm import tqdm

data_arrays = []
all_filenames = []

# First, collect all relevant directories for progress bar
join_targets = []
for root, dirs, files in os.walk('Data'):
    dirs.sort(key=lambda x: int(x) if x.isdigit() else x)
    files.sort()
    if 'data.npy' in files and 'filenames.json' in files:
        join_targets.append(root)

# Now, process with a progress bar
for root in tqdm(join_targets, desc="Joining processed data"):
    arr = np.load(os.path.join(root, 'data.npy'))
    data_arrays.append(arr)
    with open(os.path.join(root, 'filenames.json')) as f:
        names = json.load(f)
        all_filenames.extend(names)

# Concatenate all data arrays
if data_arrays:
    final_data = np.concatenate(data_arrays, axis=0)
    np.save('Data/ZH_radar_dataset.npy', final_data)
    print(f"Saved concatenated data to Data/ZH_radar_dataset.npy, shape: {final_data.shape}")
else:
    print("No data.npy files found in Data directory.")

# Save all filenames
if all_filenames:
    with open('Data/ZH_radar_filenames.json', 'w') as f:
        json.dump(all_filenames, f)
    print(f"Saved concatenated filenames to Data/ZH_radar_filenames.json, count: {len(all_filenames)}")
else:
    print("No filenames.json files found in Data directory.") 