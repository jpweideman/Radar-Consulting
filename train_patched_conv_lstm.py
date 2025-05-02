#!/usr/bin/env python
# train_patched_conv_lstm.py

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from scipy.ndimage import binary_dilation
from skimage.measure import find_contours
from matplotlib.path import Path as MplPath
import wandb
import ast

def detect_storm_mask(frame, reflectivity_threshold=45, area_threshold=15, dilation_iterations=5):
    """
    Given a single radar frame (H, W), return a boolean mask of storm regions.
    """
    mask = frame > reflectivity_threshold
    dilated = binary_dilation(mask, iterations=dilation_iterations)
    contours = find_contours(dilated.astype(float), 0.5)
    final = np.zeros_like(mask)
    for contour in contours:
        path = MplPath(contour[:, ::-1])
        xg, yg = np.meshgrid(np.arange(frame.shape[1]), np.arange(frame.shape[0]))
        pts = np.vstack((xg.ravel(), yg.ravel())).T
        inside = path.contains_points(pts).reshape(frame.shape)
        if (mask & inside).sum() >= area_threshold:
            final |= inside
    return final

class RadarPatchDataset(Dataset):
    """
    Extracts storm-centered patches from a radar cube for ConvLSTM training.

    Each sample is a seq_len_in sequence of patches predicting the next frame.
    """

    def __init__(self,
                 cube: np.ndarray,
                 seq_len_in: int,
                 seq_len_out: int,
                 patch_size: int = 64,
                 stride: int = 32,
                 reflectivity_threshold: float = 45,
                 area_threshold: int = 15,
                 dilation_iterations: int = 5):
        """
        Parameters
        ----------
        cube : ndarray
            Shape (T, C, H, W) in dBZ units.
        seq_len_in : int
            Number of input time steps.
        seq_len_out : int
            Number of output time steps (usually 1).
        patch_size : int
            Width/height of square patch.
        stride : int
            Stride for sliding patches.
        reflectivity_threshold : float
            dBZ threshold to detect storm pixels.
        area_threshold : int
            Minimum number of storm pixels per patch.
        dilation_iterations : int
            Dilation to connect nearby storm pixels.
        """
        T, C, H, W = cube.shape
        self.seq_len_in = seq_len_in
        self.seq_len_out = seq_len_out

        # Normalize to [0,1]
        cube = np.maximum(cube, 0.0)
        maxv = cube.max()
        self.maxv = float(maxv)
        eps = 1e-6
        cube_n = cube.astype(np.float32) / (maxv + eps)

        self.X, self.Y, self.coords = [], [], []

        # For each possible target index t
        for t in range(seq_len_in, T - seq_len_out + 1):
            # detect storms on the first channel of the target frame
            frame = cube[t, 0]
            storm_mask = detect_storm_mask(frame,
                                           reflectivity_threshold,
                                           area_threshold,
                                           dilation_iterations)
            if not storm_mask.any():
                continue

            # slide patches
            for i in range(0, H - patch_size + 1, stride):
                for j in range(0, W - patch_size + 1, stride):
                    if storm_mask[i:i+patch_size, j:j+patch_size].sum() >= area_threshold:
                        x_seq = cube_n[t-seq_len_in:t, :, i:i+patch_size, j:j+patch_size]
                        y_seq = cube_n[t:t+seq_len_out, :, i:i+patch_size, j:j+patch_size].squeeze(0)
                        self.X.append(x_seq)
                        self.Y.append(y_seq)
                        self.coords.append((t, i, j))

        self.X = np.stack(self.X)
        self.Y = np.stack(self.Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])  # (seq_len_in, C, H_p, W_p)
        y = torch.from_numpy(self.Y[idx])  # (C, H_p, W_p)
        coord = self.coords[idx]           # (t, i, j)
        # rearrange x to (B, S, C, H, W) will be done by DataLoader
        return x, y, coord

class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell."""

    def __init__(self, in_ch, hid_ch, kernel=3):
        super().__init__()
        assert kernel % 2 == 1, "Kernel must be odd"
        pad = kernel // 2
        self.hid_ch = hid_ch
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, kernel, padding=pad)

    def forward(self, x, h, c):
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = gates.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, B, H, W, device):
        h = torch.zeros(B, self.hid_ch, H, W, device=device)
        return h, h.clone()

class ConvLSTM(nn.Module):
    """Multi-layer ConvLSTM for spatiotemporal forecasting."""

    def __init__(self, in_ch, hidden_dims=(64,64), kernel=3):
        super().__init__()
        self.layers = nn.ModuleList()
        for idx, h in enumerate(hidden_dims):
            i_ch = in_ch if idx==0 else hidden_dims[idx-1]
            self.layers.append(ConvLSTMCell(i_ch, h, kernel))
        self.to_out = nn.Conv2d(hidden_dims[-1], in_ch, 1)

    def forward(self, x):
        # x: (B, S, C, H, W)
        B, S, C, H, W = x.shape
        device = x.device
        h_list, c_list = [], []
        for cell in self.layers:
            h, c = cell.init_hidden(B, H, W, device)
            h_list.append(h); c_list.append(c)

        for t in range(S):
            xt = x[:,t]  # (B, C, H, W)
            for i, cell in enumerate(self.layers):
                h_list[i], c_list[i] = cell(xt, h_list[i], c_list[i])
                xt = h_list[i]

        return self.to_out(xt)  # (B, C, H, W)

def reassemble_full_frames(patch_preds, coords, full_shape, patch_size):
    """
    Stitch predicted patches back into full frames by averaging overlaps.

    patch_preds: (N, C, H_p, W_p) array
    coords: list of (t, i, j)
    full_shape: (T, C, H, W)
    """
    T, C, H, W = full_shape
    full = np.zeros(full_shape, dtype=np.float32)
    count = np.zeros(full_shape, dtype=np.float32)

    for p, (t,i,j) in zip(patch_preds, coords):
        full[t,:,i:i+patch_size,j:j+patch_size] += p
        count[t,:,i:i+patch_size,j:j+patch_size] += 1

    count[count==0] = 1
    return full / count

def train_radar_model(
    npy_path: str,
    save_dir: str,
    *,
    seq_len_in: int = 10,
    seq_len_out: int = 1,
    train_frac: float = 0.8,
    batch_size: int = 4,
    lr: float = 2e-4,
    hidden_dims: tuple = (64,64),
    kernel_size: int = 3,
    epochs: int = 15,
    device: str = None,
    patch_size: int = 64,
    stride: int = 32,
    reflectivity_threshold: float = 45,
    area_threshold: int = 15,
    dilation_iterations: int = 5
):
    """
    Train ConvLSTM on storm-centered patches.

    Saves best checkpoint to save_dir/best_val.pt
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(save_dir); save_dir.mkdir(exist_ok=True, parents=True)

    cube = np.load(npy_path)
    cube[cube < 0] = 0

    # Save normalization stats for un-normalizing later
    maxv = float(cube.max())
    np.savez(save_dir/"minmax_stats.npz", maxv=maxv)

    T, C, H, W = cube.shape
    print(f"Loaded {npy_path} → {cube.shape}")

    # build dataset
    ds = RadarPatchDataset(cube,
                           seq_len_in, seq_len_out,
                           patch_size, stride,
                           reflectivity_threshold,
                           area_threshold,
                           dilation_iterations)

    # train/val split
    n = len(ds)
    n_train = int(train_frac * n)
    train_ds = Subset(ds, range(0, n_train))
    val_ds   = Subset(ds, range(n_train, n))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    print(f"Patches  train={len(train_ds)}  val={len(val_ds)}")

    # model & optimizer
    model = ConvLSTM(C, hidden_dims=hidden_dims, kernel=kernel_size).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # W&B
    wandb.init(project="radar-patch-train", config=dict(
        seq_len_in=seq_len_in, seq_len_out=seq_len_out,
        batch_size=batch_size, lr=lr,
        hidden_dims=hidden_dims, kernel_size=kernel_size,
        epochs=epochs, patch_size=patch_size,
        stride=stride, reflectivity_threshold=reflectivity_threshold,
        area_threshold=area_threshold, dilation_iterations=dilation_iterations
    ))
    wandb.watch(model)

    best_val = float("inf")
    for ep in range(1, epochs+1):
        # train
        model.train(); tot=0
        for xb, yb, _ in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * xb.size(0)
        train_loss = tot / len(train_dl.dataset)

        # val
        model.eval(); tot=0
        with torch.no_grad():
            for xb, yb, _ in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                tot += loss_fn(pred, yb).item() * xb.size(0)
        val_loss = tot / len(val_dl.dataset)

        print(f"[{ep:02d}/{epochs}] train={train_loss:.4f} val={val_loss:.4f}")
        wandb.log(dict(epoch=ep, train_loss=train_loss, val_loss=val_loss))

        # save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_dir/"best_val.pt")
            print("  ↳ new best saved")

    print("Training complete. Best val loss:", best_val)
    wandb.finish()


def predict_validation_set(
    npy_path: str,
    run_dir:  str,
    *,
    seq_len_in: int = 10,
    seq_len_out: int = 1,
    train_frac: float = 0.8,
    batch_size: int = 4,
    hidden_dims: tuple = (64,64),
    kernel_size: int = 3,
    patch_size: int = 64,
    stride: int = 32,
    device: str = None,
    save_arrays: bool = True
):
    """
    Run inference on the validation set and reassemble patches into full frames.

    Returns
    -------
    full_pred : np.ndarray
        Reconstructed full‐frame predictions (T, C, H, W) in dBZ.
    tgt_all   : np.ndarray
        Ground truth full‐frame targets (T, C, H, W) in dBZ.
    """
    import numpy as np
    import torch
    from pathlib import Path
    from torch.utils.data import DataLoader, Subset

    device = device or "cpu"
    run_dir = Path(run_dir)

    # load normalization stats
    stats = np.load(run_dir/"minmax_stats.npz")
    maxv, eps = float(stats["maxv"]), 1e-6

    # load the full cube
    cube = np.load(npy_path)
    cube[cube < 0] = 0
    T, C, H, W = cube.shape

    # rebuild the full dataset and grab coords for all samples
    from train_patched_conv_lstm import RadarPatchDataset, ConvLSTM
    ds = RadarPatchDataset(
        cube,
        seq_len_in,
        seq_len_out,
        patch_size=patch_size,
        stride=stride
    )
    all_coords = ds.coords  # list of (t,i,j) for every patch in order

    # split into val
    n_total = len(ds)
    n_train = int(n_total * train_frac)
    val_indices = list(range(n_train, n_total))
    val_coords  = [all_coords[i] for i in val_indices]

    val_ds = Subset(ds, val_indices)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # load model
    ckpt = run_dir/"best_val.pt"
    model = ConvLSTM(in_ch=C, hidden_dims=hidden_dims, kernel=kernel_size)
    st = torch.load(ckpt, map_location=device)
    # handle both plain state_dicts and dicts with 'model' key:
    if isinstance(st, dict) and "model" in st:
        state_dict = st["model"]
    else:
        state_dict = st
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # run inference on patches
    preds_p = []
    with torch.no_grad():
        for xb, _, _ in val_dl:
            xb = xb.to(device)
            out = model(xb).cpu().numpy()
            preds_p.append(out)
    preds_p = np.concatenate(preds_p, axis=0)  # (N_val, C, patch_size, patch_size)

    # reassemble into full frames
    full_pred = np.zeros((T, C, H, W), dtype=np.float32)
    count     = np.zeros_like(full_pred)
    for p, (t, i, j) in zip(preds_p, val_coords):
        full_pred[t, :, i:i+patch_size, j:j+patch_size] += p
        count    [t, :, i:i+patch_size, j:j+patch_size] += 1
    count[count == 0] = 1
    full_pred = full_pred / count

    # undo normalization to get dBZ
    full_pred = full_pred * (maxv + eps)

    # build true targets
    # target frame for each t is cube[t+seq_len_in]
    val_t = []
    for idx in val_indices:
        t, _, _ = all_coords[idx]
        val_t.append(cube[t + seq_len_in])
    tgt_all = np.stack(val_t, axis=0)  # (N_val, C, H, W)

    if save_arrays:
        np.save(run_dir/"val_preds_dBZ_full.npy", full_pred)
        np.save(run_dir/"val_targets_dBZ_full.npy", tgt_all)
        print("Saved val_preds_dBZ_full.npy + val_targets_dBZ_full.npy")

    return full_pred, tgt_all


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train or predict with patched ConvLSTM")
    p.add_argument("--npy_path",    type=str, required=True)
    p.add_argument("--save_dir",    type=str, required=True)
    p.add_argument("--hidden_dims", type=str, required=True,
                   help="e.g. \"(64,64,128)\" or \"[64,128]\"")
    p.add_argument("--kernel_size", type=int, required=True)
    p.add_argument("--seq_len_in",  type=int, default=10)
    p.add_argument("--seq_len_out", type=int, default=1)
    p.add_argument("--train_frac",  type=float, default=0.8)
    p.add_argument("--batch_size",  type=int, default=4)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--epochs",      type=int, default=15)
    p.add_argument("--device",      type=str, default="cuda")
    p.add_argument("--mode",        choices=["train","predict"], default="train")

    args = p.parse_args()

    hidden_dims = ast.literal_eval(args.hidden_dims)
    if not isinstance(hidden_dims, (tuple,list)):
        raise ValueError("hidden_dims must be tuple or list")

    if args.mode == "train":
        train_radar_model(
            args.npy_path,
            args.save_dir,
            seq_len_in=args.seq_len_in,
            seq_len_out=args.seq_len_out,
            train_frac=args.train_frac,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden_dims=hidden_dims,
            kernel_size=args.kernel_size,
            epochs=args.epochs,
            device=args.device
        )
    else:
        predict_validation_set(
            args.npy_path,
            args.save_dir,
            seq_len_in=args.seq_len_in,
            seq_len_out=args.seq_len_out,
            train_frac=args.train_frac,
            batch_size=args.batch_size,
            hidden_dims=hidden_dims,
            kernel_size=args.kernel_size,
            device=args.device
        )