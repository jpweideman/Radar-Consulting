import numpy as np
import json
import argparse
import os
from pathlib import Path
import gc

CHUNK_SIZE = 100  # Number of samples per chunk (axis 0)

def copy_chunked(src, dst, start, end, chunk_size=CHUNK_SIZE):
    """Copy slices [start:end] from src memmap to dst memmap in chunks."""
    for i in range(start, end, chunk_size):
        chunk_end = min(i + chunk_size, end)
        dst[i - start:chunk_end - start] = src[i:chunk_end]
        print(f"Copied samples {i} to {chunk_end}...")


def main():
    parser = argparse.ArgumentParser(description='Split data into first 95% and last 5%, saving both to a specified output directory (RAM/disk efficient, original is untouched).')
    parser.add_argument('--input', type=str, required=True, help='Input .npy file')
    parser.add_argument('--filenames', type=str, help='Optional: corresponding filenames .json file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save split files')
    parser.add_argument('--suffix_95', type=str, default='_first95', help='Suffix for the first 95% split (default: _first95)')
    parser.add_argument('--suffix_5', type=str, default='_last5', help='Suffix for the last 5% split (default: _last5)')
    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open memmap to original data
    arr = np.load(args.input, mmap_mode='r')
    N = arr.shape[0]
    split = int(N * 0.95)
    print(f"Total samples: {N}, first 95%: {split}, last 5%: {N-split}")

    # Prepare output paths
    orig_path = Path(args.input)
    first95_path = output_dir / (orig_path.stem + args.suffix_95 + '.npy')
    last5_path = output_dir / (orig_path.stem + args.suffix_5 + '.npy')

    # Write first 95% to new file
    print(f"Writing first 95% to: {first95_path}")
    first95_shape = (split,) + arr.shape[1:]
    first95_data = np.lib.format.open_memmap(first95_path, mode='w+', dtype=arr.dtype, shape=first95_shape)
    copy_chunked(arr, first95_data, 0, split)
    first95_data.flush()
    del first95_data
    gc.collect()

    # Write last 5% to new file
    print(f"Writing last 5% to: {last5_path}")
    last5_shape = (N - split,) + arr.shape[1:]
    last5_data = np.lib.format.open_memmap(last5_path, mode='w+', dtype=arr.dtype, shape=last5_shape)
    copy_chunked(arr, last5_data, split, N)
    last5_data.flush()
    del last5_data
    gc.collect()

    print(f"Done. First 95%: {first95_path} ({split} samples), Last 5%: {last5_path} ({N-split} samples)")

    # Filenames
    if args.filenames:
        with open(args.filenames, 'r') as f:
            names = json.load(f)
        if len(names) != N:
            raise ValueError(f"Filenames length {len(names)} does not match data samples {N}")
        first95_names = names[:split]
        last5_names = names[split:]
        first95_names_path = output_dir / (Path(args.filenames).stem + args.suffix_95 + '.json')
        last5_names_path = output_dir / (Path(args.filenames).stem + args.suffix_5 + '.json')
        with open(first95_names_path, 'w') as f:
            json.dump(first95_names, f, indent=2)
        with open(last5_names_path, 'w') as f:
            json.dump(last5_names, f, indent=2)
        print(f"Saved first 95% filenames to {first95_names_path} ({len(first95_names)})")
        print(f"Saved last 5% filenames to {last5_names_path} ({len(last5_names)})")

if __name__ == "__main__":
    main() 