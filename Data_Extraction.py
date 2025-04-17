import os
import glob
import numpy as np
import wradlib as wrl

def adjust_array_shape(arr, target_shape, mode='pad'):
    """
    Adjusts a 2D array to the target_shape.
    
    Parameters:
      - arr: Input 2D NumPy array.
      - target_shape: Desired shape tuple, e.g., (361, 240) or (360, 240).
      - mode: 'pad' (if the array is smaller, pad with zeros) or 'crop'
              (if the array is larger, crop it).
              
    Returns:
      - A new array with the target shape.
      
    Raises:
      - ValueError: if the adjustment cannot be made.
    """
    if arr.shape == target_shape:
        return arr

    current_rows, current_cols = arr.shape
    target_rows, target_cols = target_shape

    # Make sure the column sizes match (we assume only the number of rows differ).
    if current_cols != target_cols:
        raise ValueError(f"Column mismatch: got {current_cols}, expected {target_cols}")

    if mode == 'pad':
        if current_rows < target_rows:
            # Pad rows at the bottom with zeros.
            pad_rows = target_rows - current_rows
            return np.pad(arr, ((0, pad_rows), (0, 0)), mode='constant', constant_values=0)
        else:
            # If array is larger or equal in rows, return as is.
            return arr
    elif mode == 'crop':
        if current_rows > target_rows:
            # Crop the extra rows off (here, we crop from the bottom).
            return arr[:target_rows, :]
        else:
            return arr
    else:
        raise ValueError("Mode must be either 'pad' or 'crop'.")

def extract_scan0_data(data_dir, expected_vars=None, target_shape=(361,240), adjust_mode='pad'):
    """
    Extracts SCAN0 data from all HDF5 files in a directory using wradlib
    for variables: KDP, RHOHV, ZV, and ZH.
    
    Additionally, for each file the "Time" value is extracted from the metadata.
    Each variable's data is adjusted (via padding or cropping) to target_shape.
    
    Parameters:
        data_dir (str): Directory containing the .h5 files.
        expected_vars (list of str): Variables to extract (default: ["KDP", "RHOHV", "ZV", "ZH"]).
        target_shape (tuple): Desired shape for each variable's data array.
        adjust_mode (str): 'pad' to pad smaller arrays or 'crop' to crop larger arrays.
    
    Returns:
        ml_data (np.ndarray): Array of shape (num_files, num_vars, target_shape[0], target_shape[1]).
        file_names (list): List of file names processed.
        time_list (list): List of "Time" values extracted from each file's metadata.
    """
    if expected_vars is None:
        expected_vars = ["KDP", "RHOHV", "ZV", "ZH"]
    
    file_pattern = os.path.join(data_dir, "*.h5")
    file_list = glob.glob(file_pattern)
    
    all_scan0_data = []
    file_names = []
    time_list = []  # List to store the "Time" metadata for each file.
    baseline_shape = None  # To verify consistency across files.
    
    for file in file_list:
        file_basename = os.path.basename(file)
        # wradlib's utility function expects just the filename.
        full_path = wrl.util.get_wradlib_data_file(file_basename)
        print("Processing file:", full_path)
        
        try:
            data, metadata = wrl.io.read_gamic_hdf5(full_path)
        except Exception as e:
            print(f"Error reading {file_basename}: {e}")
            continue
        
        if "SCAN0" not in data:
            print(f"'SCAN0' not found in file: {file_basename}. Skipping file.")
            continue
        
        # Extract the "Time" metadata from the file.
        # Adjust the key if your metadata uses a different name.
        time_val = metadata["SCAN0"].get("Time", None)
        
        scan0 = data["SCAN0"]
        channels = []
        skip_file = False
        
        for var in expected_vars:
            if var in scan0 and "data" in scan0[var]:
                channel_data = scan0[var]["data"]
                try:
                    # Adjust the array to the target shape.
                    adjusted_data = adjust_array_shape(channel_data, target_shape, mode=adjust_mode)
                except ValueError as e:
                    print(f"Error adjusting shape for variable '{var}' in file {file_basename}: {e}. Skipping file.")
                    skip_file = True
                    break
                channels.append(adjusted_data)
            else:
                print(f"Variable '{var}' not found in SCAN0 for file {file_basename}. Skipping file.")
                skip_file = True
                break
        
        if skip_file:
            continue
        
        # Stack the variables (channels) to form an array of shape:
        # (num_vars, target_shape[0], target_shape[1])
        scan0_array = np.stack(channels, axis=0)
        print(f"File {file_basename} produced data shape {scan0_array.shape}")
        
        if baseline_shape is None:
            baseline_shape = scan0_array.shape
        elif scan0_array.shape != baseline_shape:
            print(f"File {file_basename} produced data shape {scan0_array.shape} which differs from baseline shape {baseline_shape}. Skipping file.")
            continue
        
        all_scan0_data.append(scan0_array)
        file_names.append(file_basename)
        time_list.append(time_val)  # Save the time value for this file.
    
    if all_scan0_data:
        ml_data = np.stack(all_scan0_data, axis=0)
        print("Extracted ML data shape:", ml_data.shape)
        return ml_data, file_names, time_list
    else:
        print("No valid SCAN0 data was extracted from any file.")
        return None, None, None

if __name__ == "__main__":
    # Set the WRADLIB_DATA environment variable to your data directory.
    os.environ["WRADLIB_DATA"] = "wradlib_data"
    data_directory = os.environ["WRADLIB_DATA"]
    
    # Option A: Extract padded data (target shape (361,240)).
    print("Using padding to adjust arrays to (361,240):")
    ml_data_pad, filenames_pad, times_pad = extract_scan0_data(
        data_directory, target_shape=(361,240), adjust_mode='pad'
    )
    
    # # Option B: Extract cropped data (target shape (360,240)).
    # print("\nUsing cropping to adjust arrays to (360,240):")
    # ml_data_crop, filenames_crop, times_crop = extract_scan0_data(
    #     data_directory, target_shape=(360,240), adjust_mode='crop'
    # )

    
    # Save the padded dataset along with its corresponding times in a single .npz file.
    if ml_data_pad is not None:
        print("ML data (padded) ready. Number of files processed:", ml_data_pad.shape[0])
        np.savez("Data/ml_data_pad_with_times.npz", data=ml_data_pad, times=np.array(times_pad))
        print("Saved padded dataset and times to ml_data_pad_with_times.npz")
    else:
        print("No padded data extracted.")
        
    # # Save the cropped dataset along with its corresponding times in a single .npz file.
    # if ml_data_crop is not None:
    #     print("ML data (cropped) ready. Number of files processed:", ml_data_crop.shape[0])
    #     np.savez("ml_data_crop_with_times.npz", data=ml_data_crop, times=np.array(times_crop))
    #     print("Saved cropped dataset and times to ml_data_crop_with_times.npz")
    # else:
    #     print("No cropped data extracted.")