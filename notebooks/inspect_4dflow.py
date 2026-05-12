"""
Stage 1b: Inspect OSU-MR 4D Flow .mat file structure
Confirms data loads cleanly and documents the k-space layout
"""
import numpy as np

DATA_PATH = "/home/nick_17/projects/medict/data/4D_Flow_Cartesian_Dataset_11.mat"

print("Trying scipy.io first (for older .mat format)...")
try:
    import scipy.io as sio
    data = sio.loadmat(DATA_PATH)
    keys = [k for k in data.keys() if not k.startswith('_')]
    print(f"Loaded with scipy.io. Fields: {keys}")
    for k in keys:
        arr = data[k]
        print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")
except Exception as e:
    print(f"scipy.io failed: {e}")
    print("\nTrying h5py (for newer v7.3 .mat format)...")
    import h5py
    with h5py.File(DATA_PATH, 'r') as f:
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")
        f.visititems(print_structure)
