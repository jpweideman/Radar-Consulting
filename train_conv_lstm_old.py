import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import wandb


# Dataset – raw dBZ which gets normalized by max

class RadarWindowDataset(Dataset):
    def __init__(self, cube_norm, seq_in, seq_out):
        # cube_norm: np.ndarray shape (T,C,H,W) in [0,1]
        X, Y = [], []
        last = cube_norm.shape[0] - seq_in - seq_out + 1
        for t in range(last):
            X.append(cube_norm[t:t+seq_in])
            Y.append(cube_norm[t+seq_in:t+seq_in+seq_out].squeeze(0))
        self.X = np.stack(X).astype(np.float32)  # (N,seq_in,C,H,W)
        self.Y = np.stack(Y).astype(np.float32)  # (N,C,H,W)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.Y[i])


# ConvLSTM building blocks
class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid_ch, kernel=3):
        super().__init__()
        assert kernel % 2 == 1, "Kernel size must be odd for ConvLSTM to preserve spatial dimensions!"
        p = kernel // 2
        self.hid_ch = hid_ch
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, kernel, padding=p)

    def forward(self, x, h, c):
        gates = self.conv(torch.cat([x, h], dim=1))
        i,f,o,g = gates.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g       = torch.tanh(g)
        c_next  = f * c + i * g
        h_next  = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, B, H, W, device):
        h = torch.zeros(B, self.hid_ch, H, W, device=device)
        return h, h.clone()
    
    
class ConvLSTM(nn.Module):
    def __init__(self, in_ch, hidden_dims=(64, 64), kernel=3):
        """
        in_ch: input channels
        hidden_dims: list of hidden dimensions per layer
        kernel: kernel size
        """
        super().__init__()
        self.layers = nn.ModuleList()
        for idx, h in enumerate(hidden_dims):
            i_ch = in_ch if idx == 0 else hidden_dims[idx-1]
            self.layers.append(ConvLSTMCell(i_ch, h, kernel))
        self.to_out = nn.Conv2d(hidden_dims[-1], in_ch, 1)

    def forward(self, x):
        B, S, _, H, W = x.shape
        device = x.device
        h_list, c_list = [], []
        for cell in self.layers:
            h, c = cell.init_hidden(B, H, W, device)
            h_list.append(h)
            c_list.append(c)
        for t in range(S):
            xt = x[:, t]
            for i, cell in enumerate(self.layers):
                h_list[i], c_list[i] = cell(xt, h_list[i], c_list[i])
                xt = h_list[i]
        return self.to_out(xt)


# Training function

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
    device: str = None
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
    hidden_dims : list or tuple of int, optional
        List specifying hidden channel size for each ConvLSTM layer (e.g., [32, 64, 128, 32]).
    kernel_size : int, optional
        Convolution kernel size for ConvLSTM cells (default: 3).
    epochs : int, optional
        Number of training epochs (default: 15).
    device : str, optional
        Device to run training on ('cuda' or 'cpu'); defaults to 'cuda' if available.

    Returns
    -------
    None
    """
        
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # load & sanitize
    cube = np.load(npy_path)
    cube[cube < 0] = 0
    T,C,H,W = cube.shape
    print(f"Loaded {npy_path} → {cube.shape}")

    # chronological split & min-max
    n_total = T - seq_len_in - seq_len_out + 1
    n_train = int(n_total * train_frac)
    ref = cube[:n_train+seq_len_in]
    maxv = float(ref.max())
    np.savez(save_dir/"minmax_stats.npz", maxv=maxv)
    eps = 1e-6
    cube_n = cube.astype(np.float32) / (maxv + eps)

    # DataLoaders
    full_ds  = RadarWindowDataset(cube_n, seq_len_in, seq_len_out)
    train_ds = Subset(full_ds, list(range(0, n_train)))
    val_ds   = Subset(full_ds, list(range(n_train, n_total)))
    train_dl = DataLoader(train_ds, batch_size, shuffle=False)
    val_dl   = DataLoader(val_ds,   batch_size, shuffle=False)
    print(f"Samples  train={len(train_ds)}  val={len(val_ds)}")

    # model, optimizer, loss
    model     = ConvLSTM(in_ch=C, hidden_dims=hidden_dims, kernel=kernel_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # checkpoints
    ckpt_latest = save_dir/"latest.pt"
    ckpt_best   = save_dir/"best_val.pt"
    best_val    = float('inf')
    start_ep = 1
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
        project="radar-forecasting",
        name=run_id,
        id=run_id,
        resume="allow",
        config={
            'seq_len_in': seq_len_in,
            'train_frac': train_frac,
            'batch_size': batch_size,
            'lr': lr,
            'hidden_dims': hidden_dims,
            'kernel_size': kernel_size,
            'epochs': epochs,
            'device': device
        }
    )
    wandb.watch(model)

    # training loop
    def run_epoch(dl, train=True):
        model.train() if train else model.eval()
        tot=0.0
        with torch.set_grad_enabled(train):
            for xb,yb in dl:
                xb,yb = xb.to(device), yb.to(device)
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
        torch.save({'epoch':ep,'model':model.state_dict(),
                    'optim':optimizer.state_dict(),'best_val':best_val},
                   ckpt_latest)
        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), ckpt_best)
            print("New best saved")
            wandb.log({'best_val_loss':best_val})

    print("Done. Checkpoints in", save_dir.resolve())
    wandb.finish()


# Predict entire validation set (and undoes scaling)

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
    which: str = "best",
    device: str = None,
    save_arrays: bool = True
):
    """
    Run inference on the validation set using a ConvLSTM model from train_radar_model.

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
    hidden_dims : list or tuple of int, optional
        List specifying hidden channel size for each ConvLSTM layer (e.g., [32, 64, 128, 32]).
    kernel_size : int, optional
        Convolution kernel size (default: 3).
    which : str, optional
        Which checkpoint to load: 'best' for best validation or 'latest' (default: 'best').
    device : str, optional
        Device to run inference on (default: 'cpu').
    save_arrays : bool, optional
        Whether to save predictions and targets as .npy files in run_dir (default: True).

    Returns
    -------
    pred_all : np.ndarray
        Array of shape (N, C, H, W) containing predicted radar reflectivity values.
    tgt_all : np.ndarray
        Array of shape (N, C, H, W) containing ground truth radar reflectivity values.
    """

    device = device or "cpu"
    run_dir = Path(run_dir)
    ckpt    = run_dir / ("best_val.pt" if which=="best" else "latest.pt")
    stats   = np.load(run_dir/"minmax_stats.npz")
    maxv    = float(stats['maxv']); eps=1e-6

    cube = np.load(npy_path); cube[cube<0]=0
    norm = cube.astype(np.float32)/(maxv+eps)

    T       = cube.shape[0]
    n_tot   = T - seq_len_in - seq_len_out + 1
    n_train = int(n_tot * train_frac)
    ds      = RadarWindowDataset(norm, seq_len_in, seq_len_out)
    val_ds  = Subset(ds, list(range(n_train, n_tot)))
    dl      = DataLoader(val_ds, batch_size, shuffle=False)

    model = ConvLSTM(in_ch=cube.shape[1], hidden_dims=hidden_dims, kernel=kernel_size)
    st = torch.load(ckpt, map_location=device)
    if isinstance(st, dict) and 'model' in st:
        st=st['model']
    model.load_state_dict(st)
    model.to(device).eval()

    preds, gts = [], []
    with torch.no_grad():
        for xb,yb in dl:
            xb = xb.to(device)
            out_n = model(xb).cpu().numpy()
            preds.append(out_n*(maxv+eps))
            gts.append(yb.numpy()*(maxv+eps))

    pred_all = np.concatenate(preds,axis=0)
    tgt_all  = np.concatenate(gts,axis=0)
    if save_arrays:
        np.save(run_dir/"val_preds_dBZ.npy",   pred_all)
        np.save(run_dir/"val_targets_dBZ.npy", tgt_all)
        print("Saved val_preds_dBZ.npy + val_targets_dBZ.npy →", run_dir)

    return pred_all, tgt_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ConvLSTM radar forecasting model")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save model checkpoints and stats")
    parser.add_argument("--hidden_dims", type=str, required=True, help="Hidden dimensions as tuple or list, e.g., (64, 64) or [64,64]")
    parser.add_argument("--kernel_size", type=int, required=True, help="Kernel size (must be odd number)")

    # Optional arguments
    parser.add_argument("--npy_path", type=str, default="Data/ZH_radar_dataset.npy", help="Path to input .npy radar file")
    parser.add_argument("--seq_len_in", type=int, default=10, help="Input sequence length (default: 10)")
    parser.add_argument("--seq_len_out", type=int, default=1, help="Output sequence length (default: 1)")
    parser.add_argument("--train_frac", type=float, default=0.6, help="Training fraction (default: 0.8)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (default: 4)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs (default: 15)")
    parser.add_argument("--device", type=str, default='cuda', help="Device to train on ('cuda' or 'cpu')")

    args = parser.parse_args()

    import ast
    try:
        hidden_dims = ast.literal_eval(args.hidden_dims)
        if not isinstance(hidden_dims, (tuple, list)):
            raise ValueError
    except Exception:
        raise ValueError("hidden_dims must be a tuple or list, like (64,64) or [64,64]")

    if args.kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd integer.")

    train_radar_model(
        npy_path=args.npy_path,
        save_dir=args.save_dir,
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