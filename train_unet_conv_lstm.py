import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import wandb
import ast
import os
import random
import numpy as np
import torch
from tqdm import tqdm

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Dataset – raw dBZ which gets normalized by max

class RadarWindowDataset(Dataset):
    def __init__(self, cube, seq_in, seq_out, maxv):
        # cube: np.ndarray shape (T,C,H,W) in original scale, memory-mapped
        self.cube = cube
        self.seq_in = seq_in
        self.seq_out = seq_out
        self.maxv = maxv
        self.last = cube.shape[0] - seq_in - seq_out + 1

    def __len__(self):
        return self.last

    def __getitem__(self, i):
        # Only load the required slice, clip negatives, and normalize
        X = np.maximum(self.cube[i:i+self.seq_in], 0) / (self.maxv + 1e-6)
        Y = np.maximum(self.cube[i+self.seq_in:i+self.seq_in+self.seq_out], 0) / (self.maxv + 1e-6)
        X = X.astype(np.float32)
        Y = Y.astype(np.float32).squeeze(0)
        return torch.from_numpy(X), torch.from_numpy(Y)


class PatchRadarWindowDataset(Dataset):
    def __init__(self, cube, seq_in, seq_out, maxv, patch_size=64, patch_stride=64, patch_thresh=0.4, patch_frac=0.15, patch_index_path=None):
        # cube: np.ndarray shape (T,C,H,W) in original scale, memory-mapped
        self.cube = cube
        self.seq_in = seq_in
        self.seq_out = seq_out
        self.maxv = maxv
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.patch_thresh = patch_thresh
        self.patch_frac = patch_frac
        self.patches = []  # List of (t, y, x)
        T, C, H, W = cube.shape
        last = T - seq_in - seq_out + 1
        # Patch index caching
        if patch_index_path is not None and os.path.exists(patch_index_path):
            print(f"Loading patch indices from {patch_index_path}")
            self.patches = np.load(patch_index_path, allow_pickle=True).tolist()
        else:
            for t in tqdm(range(last), desc='Extracting patches'):
                for y in range(0, H - patch_size + 1, patch_stride):
                    for x in range(0, W - patch_size + 1, patch_stride):
                        Y_patch = np.maximum(
                            cube[t+seq_in:t+seq_in+seq_out, :, y:y+patch_size, x:x+patch_size], 0
                        ) / (maxv + 1e-6)
                        total_pix = Y_patch.size
                        n_above = (Y_patch > patch_thresh).sum()
                        if n_above / total_pix >= patch_frac:
                            self.patches.append((t, y, x))
            if patch_index_path is not None:
                np.save(patch_index_path, np.array(self.patches, dtype=object))
                print(f"Saved patch indices to {patch_index_path}")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, i):
        t, y, x = self.patches[i]
        X_patch = np.maximum(
            self.cube[t:t+self.seq_in, :, y:y+self.patch_size, x:x+self.patch_size], 0
        ) / (self.maxv + 1e-6)
        Y_patch = np.maximum(
            self.cube[t+self.seq_in:t+self.seq_in+self.seq_out, :, y:y+self.patch_size, x:x+self.patch_size], 0
        ) / (self.maxv + 1e-6)
        X_patch = X_patch.astype(np.float32)
        Y_patch = Y_patch.astype(np.float32).squeeze(0)
        return torch.from_numpy(X_patch), torch.from_numpy(Y_patch), t, y, x


# ConvLSTM building blocks
class DoubleConv(nn.Module):
    """(Conv2d => ReLU) * 2"""
    def __init__(self, in_ch, out_ch, kernel=3):
        super().__init__()
        p = kernel // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=p),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel, padding=p),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_ch, out_ch, kernel=3):
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, kernel)
        )

    def forward(self, x):
        return self.mpconv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_ch, skip_ch, out_ch, kernel=3):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch, kernel)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid_ch, kernel=3):
        super().__init__()
        p = kernel // 2
        self.hid_ch = hid_ch
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, kernel, padding=p)

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

class UNetConvLSTM(nn.Module):
    """
    U-Net + ConvLSTM model for spatiotemporal prediction.

    Parameters
    ----------
    in_ch : int
        Number of input channels.
    out_ch : int
        Number of output channels.
    base_ch : int
        Number of channels in the first encoder layer.
    lstm_hid : int or tuple/list of int
        Number of hidden channels in the ConvLSTM bottleneck.
        If a tuple/list, multiple ConvLSTM layers are stacked.
    seq_len : int
        Input sequence length (number of time steps).
    kernel : int
        Convolution kernel size for all convolutions (must be odd).
    """
    def __init__(self, in_ch, out_ch, base_ch=32, lstm_hid=64, seq_len=10, kernel=3):
        super().__init__()
        self.seq_len = seq_len
        self.kernel = kernel
        # Encoder
        self.inc = DoubleConv(in_ch, base_ch, kernel)
        self.down1 = Down(base_ch, base_ch*2, kernel)
        self.down2 = Down(base_ch*2, base_ch*4, kernel)
        # Bottleneck ConvLSTM (support multiple layers if lstm_hid is tuple/list)
        if isinstance(lstm_hid, (tuple, list)):
            self.lstm_layers = nn.ModuleList()
            in_dim = base_ch*4
            for hid in lstm_hid:
                self.lstm_layers.append(ConvLSTMCell(in_dim, hid, kernel))
                in_dim = hid
            self.lstm_out_dim = lstm_hid[-1]
        else:
            self.lstm_layers = None
            self.lstm = ConvLSTMCell(base_ch*4, lstm_hid, kernel)
            self.lstm_out_dim = lstm_hid
        # Decoder
        self.up1 = Up(self.lstm_out_dim, base_ch*2, base_ch*2, kernel)
        self.up2 = Up(base_ch*2, base_ch, base_ch, kernel)
        self.outc = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, x):
        # x: (B, S, C, H, W)
        B, S, C, H, W = x.shape
        device = x.device
        # Encode each frame, then stack
        x1_seq, x2_seq, x3_seq = [], [], []
        for t in range(S):
            x1 = self.inc(x[:, t])
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x1_seq.append(x1)
            x2_seq.append(x2)
            x3_seq.append(x3)
        # Stack over time
        x3_seq = torch.stack(x3_seq, dim=1)  # (B, S, C, H, W)
        # ConvLSTM over bottleneck
        if self.lstm_layers is not None:
            x_lstm = x3_seq  # (B, S, C, H, W)
            for layer in self.lstm_layers:
                h, c = layer.init_hidden(B, x_lstm.size(-2), x_lstm.size(-1), device)
                outputs = []
                for t in range(S):
                    h, c = layer(x_lstm[:, t], h, c)
                    outputs.append(h)
                x_lstm = torch.stack(outputs, dim=1)  # (B, S, C, H, W)
            h = x_lstm[:, -1]  # Use the last time step's output for decoding
        else:
            h, c = self.lstm.init_hidden(B, x3_seq.size(-2), x3_seq.size(-1), device)
            for t in range(S):
                h, c = self.lstm(x3_seq[:, t], h, c)
        # Decoder with skip connections (use last encoder features)
        x = self.up1(h, x2_seq[-1])
        x = self.up2(x, x1_seq[-1])
        x = self.outc(x)
        return x


# Weighted MSE loss

def weighted_mse_loss(pred, target, threshold=0.40, weight_high=10.0):
    """
    Weighted MSE that emphasizes high reflectivity areas (e.g., >40 dBZ in original scale).
    Assumes pred and target are normalized to [0,1].

    Parameters:
    -----------
    threshold: float
        Normalized reflectivity threshold (e.g., 0.40 for normalized reflectivity between 0 and 1).
    weight_high: float
        Weight multiplier for pixels where true > threshold.
    """
    weight = torch.ones_like(target)
    weight[target > threshold] = weight_high
    return ((pred - target) ** 2 * weight).mean()


# Training function

def atomic_save(obj, path):
    tmp_path = str(path) + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)

# Instead of computing maxv from training data, we use a fixed value of 85. Almost all the data is below 85.           
# def compute_maxv(cube, end_idx, chunk_size=100):
#     maxv = 0.0
#     for i in range(0, end_idx, chunk_size):
#         chunk = cube[i:min(i+chunk_size, end_idx)]
#         chunk_max = np.max(np.maximum(chunk, 0))
#         if chunk_max > maxv:
#             maxv = chunk_max
#     return float(maxv)

def train_radar_model(
    npy_path: str,
    save_dir: str,
    *,
    seq_len_in: int = 10,
    seq_len_out: int = 1,
    train_frac: float = 0.8,
    batch_size: int = 4,
    lr: float = 2e-4,
    kernel: int = 3,
    epochs: int = 15,
    device: str = "cuda" ,
    loss_name: str = "mse",
    loss_weight_thresh: float = 0.40,
    loss_weight_high: float = 10.0,
    patch_size: int = 64,
    patch_stride: int = 64,
    patch_thresh: float = 0.4,
    patch_frac: float = 0.15,
    use_patches: bool = False,
    base_ch: int = 32,
    lstm_hid: int = 64,
    wandb_project: str = "radar-forecasting",
):
    """
    Train a ConvLSTM radar forecasting model.

    Parameters
    ----------
    npy_path : str
        Path to the input NumPy file containing radar reflectivity data with shape (T, C, H, W).
    save_dir : str
        Directory to save model checkpoints and statistics.
    seq_len_in : int, optional
        Number of input time steps (default: 10).
    seq_len_out : int, optional
        Number of output time steps to predict (default: 1).
    train_frac : float, optional
        Fraction of the data to use for training; the remainder is used for validation (default: 0.8).
    batch_size : int, optional
        Batch size for training (default: 4).
    lr : float, optional
        Learning rate for the optimizer (default: 2e-4).
    kernel : int, optional
        Convolution kernel size for all convolutions (default: 3, must be odd).
    epochs : int, optional
        Number of training epochs (default: 15).
    device : str, optional
        Device to run training on ('cuda' or 'cpu'); defaults to 'cuda' if available.
    loss_name : str, optional
        Loss function to use; either 'mse', 'weighted_mse'.
    loss_weight_thresh : float, optional (used for weighted_mse and masked_mse)
        Normalized reflectivity threshold (e.g., 0.40 for normalized reflectivity between 0 and 1. Equivalent to 40 dBZ in original scale).
    loss_weight_high : float, optional (used for weighted_mse)
        Weight multiplier for pixels where true > threshold.
    patch_size : int, optional
        Size of spatial patches to extract (default: 64).
    patch_stride : int, optional
        Stride for patch extraction (default: 64).
    patch_thresh : float, optional
        Threshold for extracting patches (default: 0.4).
    patch_frac : float, optional
        Minimum fraction of pixels in patch above threshold (default: 0.15).
    use_patches : bool, optional
        Whether to use patch-based training (default: False).
    base_ch : int, optional
        Base number of channels for U-Net (default: 32).
    lstm_hid : int or tuple/list of int, optional
        Number of hidden channels in the ConvLSTM bottleneck (default: 64).
        If a tuple or list is provided, multiple ConvLSTM layers are stacked in the bottleneck,
        with each value specifying the hidden size of each layer.
    wandb_project : str, optional
        wandb project name (default: "radar-forecasting").

    Returns
    -------
    None
    """
        
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # load & sanitize (memory-mapped)
    cube = np.load(npy_path, mmap_mode='r')
    T,C,H,W = cube.shape
    print(f"Loaded {npy_path} → {cube.shape}")

    # chronological split & min-max
    n_total = T - seq_len_in - seq_len_out + 1
    n_train = int(n_total * train_frac)
    n_train_plus = n_train + seq_len_in
    # maxv = compute_maxv(cube, n_train_plus, chunk_size=100)^
    maxv = 85.0
    print(f"Normalization maxv (fixed): {maxv}")
    np.savez(save_dir/"minmax_stats.npz", maxv=maxv)
    eps = 1e-6

    # DataLoaders
    if use_patches:
        patch_index_path = str(save_dir / "patch_indices.npy")
        full_ds  = PatchRadarWindowDataset(cube, seq_len_in, seq_len_out, maxv, patch_size, patch_stride, patch_thresh, patch_frac, patch_index_path=patch_index_path)
        # Split by time index (t) for train/val
        train_idx = []
        val_idx = []
        n_total = T - seq_len_in - seq_len_out + 1
        n_train = int(n_total * train_frac)
        for i, (t, y, x) in enumerate(full_ds.patches):
            if t < n_train:
                train_idx.append(i)
            else:
                val_idx.append(i)
        train_ds = Subset(full_ds, train_idx)
        val_ds   = Subset(full_ds, val_idx)
        train_dl = DataLoader(train_ds, batch_size, shuffle=True)
        val_dl   = DataLoader(val_ds,   batch_size, shuffle=False)
        print(f"Patch-based: train={len(train_ds)}  val={len(val_ds)}")
    else:
        full_ds  = RadarWindowDataset(cube, seq_len_in, seq_len_out, maxv)
        train_ds = Subset(full_ds, list(range(0, n_train)))
        val_ds   = Subset(full_ds, list(range(n_train, n_total)))
        train_dl = DataLoader(train_ds, batch_size, shuffle=False)
        val_dl   = DataLoader(val_ds,   batch_size, shuffle=False)
        print(f"Samples  train={len(train_ds)}  val={len(val_ds)}")

    # model, optimizer, loss
    model     = UNetConvLSTM(
        in_ch=C,
        out_ch=C,
        base_ch=base_ch,
        lstm_hid=lstm_hid,
        seq_len=seq_len_in,
        kernel=kernel
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if loss_name == "mse":
        criterion = nn.MSELoss()
    elif loss_name == "weighted_mse":
        criterion = lambda pred, tgt: weighted_mse_loss(
            pred, tgt,
            threshold=loss_weight_thresh,
            weight_high=loss_weight_high
        )
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")

    # checkpoints
    ckpt_latest = save_dir/"latest.pt"
    ckpt_best   = save_dir/"best_val.pt"
    best_val    = float('inf')
    start_ep = 1
    epochs_since_improvement = 0
    if ckpt_latest.exists():
        st = torch.load(ckpt_latest, map_location=device)
        model.load_state_dict(st['model'])
        optimizer.load_state_dict(st['optim'])
        best_val = st['best_val']
        start_ep = st['epoch'] + 1
        print(f"✔ Resumed epoch {st['epoch']} (best_val={best_val:.4f})")

    end_epoch = start_ep + epochs - 1

    # wandb
    run_id = save_dir.name
    wandb.init(
        project=wandb_project,
        name=run_id,
        id=run_id,
        resume="allow",
        config={
            'seq_len_in': seq_len_in,
            'train_frac': train_frac,
            'batch_size': batch_size,
            'lr': lr,
            'kernel': kernel,
            'epochs': epochs,
            'device': device,
            'loss_name': loss_name,
            'loss_weight_thresh': loss_weight_thresh,
            'loss_weight_high': loss_weight_high,
            'patch_size': patch_size,
            'patch_stride': patch_stride,
            'patch_thresh': patch_thresh,
            'patch_frac': patch_frac,
            'use_patches': use_patches,
            'base_ch': base_ch,
            'lstm_hid': lstm_hid
        }
    )
    wandb.watch(model)

    # training loop
    def run_epoch(dl, train=True):
        model.train() if train else model.eval()
        tot=0.0
        with torch.set_grad_enabled(train):
            for batch in tqdm(dl, desc=("Train" if train else "Val"), leave=False):
                if use_patches:
                    xb, yb = batch[0], batch[1]  # ignore t, y, x
                else:
                    xb, yb = batch
                xb, yb = xb.to(device), yb.to(device)
                pred  = model(xb)
                loss  = criterion(pred, yb)
                if train:
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                tot += loss.item()*xb.size(0)
        return tot/len(dl.dataset)

    for ep in range(start_ep, end_epoch+1):
        tr = run_epoch(train_dl, True)
        vl = run_epoch(val_dl,   False)
        print(f"[{ep:02d}/{end_epoch}] train {tr:.4f} | val {vl:.4f}")
        wandb.log({'epoch':ep,'train_loss':tr,'val_loss':vl})
        atomic_save({'epoch':ep,'model':model.state_dict(),
                    'optim':optimizer.state_dict(),'best_val':best_val},
                   ckpt_latest)
        if vl < best_val:
            best_val = vl
            atomic_save(model.state_dict(), ckpt_best)
            print("New best saved")
            wandb.log({'best_val_loss':best_val})
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
        if epochs_since_improvement >= 10:
            print(f"Early stopping: validation loss did not improve for {epochs_since_improvement} epochs.")
            break

    print("Done. Checkpoints in", save_dir.resolve())
    wandb.finish()


def compute_mse_by_ranges(pred, target, ranges):
    """
    Compute MSE for different reflectivity ranges.
    
    Parameters:
    -----------
    pred: np.ndarray
        Predicted values
    target: np.ndarray
        Ground truth values
    ranges: list of tuples
        List of (min, max) ranges to compute MSE for
        
    Returns:
    --------
    dict
        Dictionary with MSE values for each range
    """
    mse_by_range = {}
    for r_min, r_max in ranges:
        mask = (target >= r_min) & (target < r_max)
        if np.any(mask):
            mse = np.mean((pred[mask] - target[mask]) ** 2)
            mse_by_range[f"mse_{r_min}_{r_max}"] = mse
        else:
            mse_by_range[f"mse_{r_min}_{r_max}"] = np.nan
    return mse_by_range


def predict_validation_set(
    npy_path: str,
    run_dir:  str,
    *,
    seq_len_in: int = 10,
    seq_len_out: int = 1,
    train_frac: float = 0.8,
    batch_size: int = 4,
    kernel: int = 3,
    which: str = "best",
    device: str = None,
    save_arrays: bool = True,
    base_ch: int = 32,
    lstm_hid: int = 64,
):
    """
    Run inference on the validation set using a U-Net+ConvLSTM model from train_radar_model.

    Parameters
    ----------
    npy_path : str
        Path to the input NumPy file containing radar reflectivity data with shape (T, C, H, W).
    run_dir : str
        Directory containing model checkpoints and statistics.
    seq_len_in : int, optional
        Number of input time steps (default: 10).
    seq_len_out : int, optional
        Number of output time steps to predict (default: 1).
    train_frac : float, optional
        Fraction of data used for training split (default: 0.8).
    batch_size : int, optional
        Batch size for inference (default: 4).
    kernel : int, optional
        Convolution kernel size (default: 3).
    which : str, optional
        Which checkpoint to load: 'best' for best validation or 'latest' (default: 'best').
    device : str, optional
        Device to run inference on (default: 'cpu').
    save_arrays : bool, optional
        Whether to save predictions and targets as .npy files in run_dir (default: True).
    base_ch : int, optional
        Base number of channels for U-Net (default: 32).
    lstm_hid : int or tuple/list of int, optional
        Number of hidden channels in the ConvLSTM bottleneck (default: 64).
        If a tuple or list is provided, multiple ConvLSTM layers are stacked in the bottleneck,
        with each value specifying the hidden size of each layer.

    Returns
    -------
    None
        This function does not return arrays. Instead, it saves the predictions and targets as memmap .npy files
        (val_preds_dBZ.npy, val_targets_dBZ.npy) and saves their shape and dtype as .npz metadata files
        (val_preds_dBZ_meta.npz, val_targets_dBZ_meta.npz) in the run_dir for later loading.
    """

    import numpy as np

    device = device or "cpu"
    run_dir = Path(run_dir)
    ckpt    = run_dir / ("best_val.pt" if which=="best" else "latest.pt")
    stats   = np.load(run_dir/"minmax_stats.npz")
    maxv    = float(stats['maxv']); eps=1e-6

    # Use memory-mapped loading for large datasets
    cube = np.load(npy_path, mmap_mode='r')
    T, C, H, W = cube.shape
    n_tot   = T - seq_len_in - seq_len_out + 1
    n_train = int(n_tot * train_frac)
    ds      = RadarWindowDataset(cube, seq_len_in, seq_len_out, maxv)
    val_ds  = Subset(ds, list(range(n_train, n_tot)))
    dl      = DataLoader(val_ds, batch_size, shuffle=False)

    model = UNetConvLSTM(
        in_ch=C,
        out_ch=C,
        base_ch=base_ch,
        lstm_hid=lstm_hid,
        seq_len=seq_len_in,
        kernel=kernel
    )
    st = torch.load(ckpt, map_location=device)
    if isinstance(st, dict) and 'model' in st:
        st=st['model']
    model.load_state_dict(st)
    model.to(device).eval()

    N = len(val_ds)
    if save_arrays:
        preds_memmap = np.memmap(run_dir/"val_preds_dBZ.npy", dtype='float32', mode='w+', shape=(N, C, H, W))
        gts_memmap   = np.memmap(run_dir/"val_targets_dBZ.npy", dtype='float32', mode='w+', shape=(N, C, H, W))
    else:
        preds_memmap = None
        gts_memmap = None

    # For MSE by range, accumulate sum of squared errors and counts for each range
    ranges = [(0, 20), (20, 35), (35, 45), (45, 100)]
    mse_sums = {f"mse_{r_min}_{r_max}": 0.0 for r_min, r_max in ranges}
    mse_counts = {f"mse_{r_min}_{r_max}": 0 for r_min, r_max in ranges}

    idx = 0
    with torch.no_grad():
        for xb, yb in tqdm(dl, desc='Validating', total=len(dl)):
            xb = xb.to(device)
            out_n = model(xb).cpu().numpy()  # (B, C, H, W)
            yb_np = yb.numpy()  # (B, C, H, W)
            out_n_dBZ = out_n * (maxv+eps)
            yb_dBZ = yb_np * (maxv+eps)
            batch_size = out_n.shape[0]
            if save_arrays:
                preds_memmap[idx:idx+batch_size] = out_n_dBZ
                gts_memmap[idx:idx+batch_size] = yb_dBZ
            # Compute MSE by range for this batch
            for r_min, r_max in ranges:
                mask = (yb_dBZ >= r_min) & (yb_dBZ < r_max)
                n_pix = np.sum(mask)
                if n_pix > 0:
                    mse = np.sum((out_n_dBZ[mask] - yb_dBZ[mask]) ** 2)
                    mse_sums[f"mse_{r_min}_{r_max}"] += mse
                    mse_counts[f"mse_{r_min}_{r_max}"] += n_pix
            idx += batch_size

    if save_arrays:
        preds_memmap.flush()
        gts_memmap.flush()
        # Check file integrity
        try:
            arr = np.load(run_dir/'val_preds_dBZ.npy', mmap_mode='r')
            print('val_preds_dBZ.npy loaded successfully:', arr.shape)
        except Exception as e:
            print('Error loading val_preds_dBZ.npy:', e)
        # Save shape and dtype metadata for memmap arrays
        meta = {
            'shape': (N, C, H, W),
            'dtype': 'float32'
        }
        np.savez(run_dir/"val_preds_dBZ_meta.npz", **meta)
        np.savez(run_dir/"val_targets_dBZ_meta.npz", **meta)

    # Finalize MSE by range
    mse_by_range = {}
    for r_min, r_max in ranges:
        key = f"mse_{r_min}_{r_max}"
        if mse_counts[key] > 0:
            mse_by_range[key] = mse_sums[key] / mse_counts[key]
        else:
            mse_by_range[key] = np.nan
    
    # Save MSE metrics
    np.savez(run_dir/"mse_by_range.npz", **mse_by_range)
    print("MSE by reflectivity range:")
    for range_name, mse in mse_by_range.items():
        print(f"{range_name}: {mse:.4f}")
    
    if save_arrays:
        print("Saved val_preds_dBZ.npy + val_targets_dBZ.npy →", run_dir)

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or validate a U-Net+ConvLSTM radar forecasting model.")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # Subparser for training
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--save_dir", type=str, required=True, help="Directory to save model checkpoints and stats")
    train_parser.add_argument("--kernel", type=int, default=3, help="Kernel size for all convolutions (default: 3, must be odd)")
    train_parser.add_argument("--npy_path", type=str, default="Data/ZH_radar_dataset.npy", help="Path to input .npy radar file")
    train_parser.add_argument("--seq_len_in", type=int, default=10, help="Input sequence length (default: 10)")
    train_parser.add_argument("--seq_len_out", type=int, default=1, help="Output sequence length (default: 1)")
    train_parser.add_argument("--train_frac", type=float, default=0.6, help="Training fraction (default: 0.8)")
    train_parser.add_argument("--batch_size", type=int, default=1, help="Batch size (default: 4)")
    train_parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    train_parser.add_argument("--epochs", type=int, default=15, help="Number of epochs (default: 15)")
    train_parser.add_argument("--device", type=str, default='cuda', help="Device to train on ('cuda' or 'cpu')")
    train_parser.add_argument("--loss_name", type=str, default="mse", help="Loss function: mse or weighted_mse")
    train_parser.add_argument("--loss_weight_thresh", type=float, default=0.35,
                    help="Threshold in normalized space to apply higher loss weighting or masking (default: 0.40)")
    train_parser.add_argument("--loss_weight_high", type=float, default=10.0,
                        help="Weight multiplier for pixels above threshold (default: 10.0)")
    train_parser.add_argument("--patch_size", type=int, default=64, help="Size of spatial patches to extract (default: 64)")
    train_parser.add_argument("--patch_stride", type=int, default=32, help="Stride for patch extraction (default: 32)")
    train_parser.add_argument("--patch_thresh", type=float, default=0.35, help="Threshold for extracting patches (default: 0.4)")
    train_parser.add_argument("--patch_frac", type=float, default=0.05, help="Minimum fraction of pixels in patch above threshold (default: 0.05)")
    train_parser.add_argument("--use_patches", type=bool, default=True, help="Whether to use patch-based training (default: True)")
    train_parser.add_argument("--base_ch", type=int, default=32, help="Base number of channels for U-Net (default: 32)")
    train_parser.add_argument("--lstm_hid", type=str, default="64", help="Number of hidden channels in the ConvLSTM bottleneck (int or tuple/list, e.g., 64 or (64,128))")
    train_parser.add_argument("--wandb_project", type=str, default="radar-forecasting", help="wandb project name")

    # Subparser for validation
    val_parser = subparsers.add_parser("validate", help="Run validation and compute MSE by reflectivity range")
    val_parser.add_argument("--npy_path", type=str, required=True, help="Path to input .npy radar file")
    val_parser.add_argument("--run_dir", type=str, required=True, help="Directory containing model checkpoints and stats")
    val_parser.add_argument("--seq_len_in", type=int, default=10, help="Input sequence length (default: 10)")
    val_parser.add_argument("--seq_len_out", type=int, default=1, help="Output sequence length (default: 1)")
    val_parser.add_argument("--train_frac", type=float, default=0.6, help="Training fraction (default: 0.8)")
    val_parser.add_argument("--batch_size", type=int, default=4, help="Batch size (default: 4)")
    val_parser.add_argument("--kernel", type=int, default=3, help="Kernel size for all convolutions (default: 3)")
    val_parser.add_argument("--which", type=str, default="best", help="Which checkpoint to load: 'best' or 'latest'")
    val_parser.add_argument("--device", type=str, default=None, help="Device to run inference on (default: 'cpu')")
    val_parser.add_argument("--save_arrays", type=bool, default=True, help="Whether to save predictions and targets as .npy files")
    val_parser.add_argument("--base_ch", type=int, default=32, help="Base number of channels for U-Net (default: 32)")
    val_parser.add_argument("--lstm_hid", type=str, default="64", help="Number of hidden channels in the ConvLSTM bottleneck (int or tuple/list, e.g., 64 or (64,128))")

    args = parser.parse_args()

    if args.command == "train":
        try:
            if isinstance(args.lstm_hid, str):
                lstm_hid = ast.literal_eval(args.lstm_hid)
            else:
                lstm_hid = args.lstm_hid
        except Exception:
            raise ValueError("lstm_hid must be an int or tuple/list, like 64 or (64,128)")

        if args.kernel % 2 == 0:
            raise ValueError("kernel must be an odd integer.")

        train_radar_model(
            npy_path=args.npy_path,
            save_dir=args.save_dir,
            seq_len_in=args.seq_len_in,
            seq_len_out=args.seq_len_out,
            train_frac=args.train_frac,
            batch_size=args.batch_size,
            lr=args.lr,
            kernel=args.kernel,
            epochs=args.epochs,
            device=args.device,
            loss_name=args.loss_name,
            loss_weight_thresh=args.loss_weight_thresh,
            loss_weight_high=args.loss_weight_high,
            patch_size=args.patch_size,
            patch_stride=args.patch_stride,
            patch_thresh=args.patch_thresh,
            patch_frac=args.patch_frac,
            use_patches=args.use_patches,
            base_ch=args.base_ch,
            lstm_hid=lstm_hid,
            wandb_project=args.wandb_project,
        )
    elif args.command == "validate":
        try:
            if isinstance(args.lstm_hid, str):
                lstm_hid = ast.literal_eval(args.lstm_hid)
            else:
                lstm_hid = args.lstm_hid
        except Exception:
            raise ValueError("lstm_hid must be an int or tuple/list, like 64 or (64,128)")
        predict_validation_set(
            npy_path=args.npy_path,
            run_dir=args.run_dir,
            seq_len_in=args.seq_len_in,
            seq_len_out=args.seq_len_out,
            train_frac=args.train_frac,
            batch_size=args.batch_size,
            kernel=args.kernel,
            which=args.which,
            device=args.device,
            save_arrays=args.save_arrays,
            base_ch=args.base_ch,
            lstm_hid=lstm_hid,
        )
