import sys
from pathlib import Path
# Add the project root directory to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import wandb  # Add wandb import
import argparse
import os
import random
from tqdm import tqdm

from src.models.conv_lstm_nonnormalized import ConvLSTM

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def atomic_save(obj, path):
    tmp_path = str(path) + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)
# Dataset – raw dBZ without normalizatio
class RadarWindowDataset(Dataset):
    def __init__(self, cube, seq_in, seq_out):
        # cube: np.ndarray shape (T,C,H,W) in original dBZ scale
        X, Y = [], []
        last = cube.shape[0] - seq_in - seq_out + 1
        for t in range(last):
            X.append(cube[t:t+seq_in])
            Y.append(cube[t+seq_in:t+seq_in+seq_out].squeeze(0))
        self.X = np.stack(X).astype(np.float32)  # (N,seq_in,C,H,W)
        self.Y = np.stack(Y).astype(np.float32)  # (N,C,H,W)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.Y[i])

# Training function
def weighted_mse_loss(pred, target, threshold=40.0, weight_high=10.0):
    """
    Weighted MSE loss in dBZ units, emphasizing high-reflectivity areas.
    pred, target: dBZ units (not normalized).
    threshold: dBZ value above which to apply weight_high (e.g., 40.0)
    weight_high: weight for pixels above threshold
    """
    weight = torch.ones_like(target)
    weight[target > threshold] = weight_high
    return ((pred - target) ** 2 * weight).mean()

def train_radar_model(
    npy_path: str,
    save_dir: str,
    args,
    *,
    seq_len_in: int = 10,
    seq_len_out: int = 1,
    train_val_test_split: tuple = (0.7, 0.15, 0.15),
    batch_size: int = 4,
    lr: float = 2e-4,
    hidden_dims: tuple = (64,64),
    kernel_size: int = 3,
    epochs: int = 15,
    device: str = "cuda" ,
    loss_name: str = "mse",
    loss_weight_thresh: float = 40.0,
    loss_weight_high: float = 10.0,
    wandb_project: str = "radar-forecasting",
    early_stopping_patience: int = 10,
):
    """
    Train a ConvLSTM radar forecasting model (no normalization).

    Pass the --no_wandb argument to disable wandb logging during training.

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
    train_val_test_split : tuple, optional
        Tuple/list of three floats (train, val, test) that sum to 1.0 (default: (0.7, 0.15, 0.15)).
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
    loss_name : str, optional
        Loss function to use; either 'mse' or 'weighted_mse' (default: 'mse').
    loss_weight_thresh : float, optional (used for weighted_mse)
        Reflectivity threshold in dBZ (e.g., 40.0).
    loss_weight_high : float, optional (used for weighted_mse)
        Weight multiplier for pixels where true > threshold.
    wandb_project : str, optional
        Weights & Biases project name for experiment tracking.
    early_stopping_patience : int, optional
        Number of epochs with no improvement before early stopping (default: 10).

    Returns
    -------
    None
    """
    if not (isinstance(train_val_test_split, (tuple, list)) and len(train_val_test_split) == 3):
        raise ValueError("train_val_test_split must be a tuple/list of three floats (train, val, test)")
    if not abs(sum(train_val_test_split) - 1.0) < 1e-6:
        raise ValueError(f"train_val_test_split must sum to 1.0, got {train_val_test_split} (sum={sum(train_val_test_split)})")
    train_frac, val_frac, _ = train_val_test_split

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # load & sanitize
    cube = np.load(npy_path)
    cube[cube < 0] = 0
    T,C,H,W = cube.shape
    print(f"Loaded {npy_path} → {cube.shape}")

    # chronological split
    n_total = T - seq_len_in - seq_len_out + 1
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)
    idx_train = list(range(0, n_train))
    idx_val = list(range(n_train, n_train + n_val))

    # DataLoaders
    full_ds  = RadarWindowDataset(cube, seq_len_in, seq_len_out)
    train_ds = Subset(full_ds, idx_train)
    val_ds   = Subset(full_ds, idx_val)
    train_dl = DataLoader(train_ds, batch_size, shuffle=False)
    val_dl   = DataLoader(val_ds,   batch_size, shuffle=False)
    print(f"Samples  train={len(train_ds)}  val={len(val_ds)}")

    # model, optimizer, loss
    model     = ConvLSTM(in_ch=C, hidden_dims=hidden_dims, kernel=kernel_size).to(device)
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
    if ckpt_latest.exists():
        st = torch.load(ckpt_latest, map_location=device)
        model.load_state_dict(st['model'])
        optimizer.load_state_dict(st['optim'])
        best_val = st['best_val']
        start_ep = st['epoch'] + 1
        print(f"✔ Resumed epoch {st['epoch']} (best_val={best_val:.4f})")

    end_epoch = start_ep + epochs - 1

    # wandb
    if not args.no_wandb:
        run_id = save_dir.name
        wandb.init(
            project=wandb_project,
            name=run_id,
            id=run_id,
            resume="allow",
            dir="experiments/wandb",
            config={
                'seq_len_in': seq_len_in,
                'seq_len_out': seq_len_out,
                'train_val_test_split': train_val_test_split,
                'batch_size': batch_size,
                'lr': lr,
                'hidden_dims': hidden_dims,
                'kernel_size': kernel_size,
                'epochs': epochs,
                'device': device,
                'loss_name': loss_name,
                'loss_weight_thresh': loss_weight_thresh,
                'loss_weight_high': loss_weight_high,
                'wandb_project': wandb_project,
                'early_stopping_patience': early_stopping_patience
            }
        )
        wandb.watch(model)

    # training loop
    def run_epoch(dl, train=True):
        model.train() if train else model.eval()
        tot=0.0
        with torch.set_grad_enabled(train):
            for batch in tqdm(dl, desc=("Train" if train else "Val"), leave=False):
                xb, yb = batch
                xb, yb = xb.to(device), yb.to(device)
                pred  = model(xb)
                loss  = criterion(pred, yb)
                if train:
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                tot += loss.item()*xb.size(0)
        return tot/len(dl.dataset)

    epochs_since_improvement = 0
    for ep in range(start_ep, end_epoch+1):
        tr = run_epoch(train_dl, True)
        vl = run_epoch(val_dl,   False)
        print(f"[{ep:02d}/{end_epoch}] train {tr:.4f} | val {vl:.4f}")
        if not args.no_wandb:
            wandb.log({'epoch':ep,'train_loss':tr,'val_loss':vl})
        atomic_save({'epoch':ep,'model':model.state_dict(),
                    'optim':optimizer.state_dict(),'best_val':best_val},
                   ckpt_latest)
        if vl < best_val:
            best_val = vl
            atomic_save(model.state_dict(), ckpt_best)
            print("New best saved")
            if not args.no_wandb:
                wandb.log({'best_val_loss':best_val})
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
        # Only apply early stopping if patience > 0
        if early_stopping_patience > 0 and epochs_since_improvement >= early_stopping_patience:
            print(f"Early stopping: validation loss did not improve for {epochs_since_improvement} epochs.")
            break

    print("Done. Checkpoints in", save_dir.resolve())
    if not args.no_wandb:
        wandb.finish()

# predict_validation_set  
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

def predict_test_set(
    npy_path: str,
    run_dir:  str,
    *,
    seq_len_in: int = 10,
    seq_len_out: int = 1,
    train_val_test_split: tuple = (0.7, 0.15, 0.15),
    batch_size: int = 4,
    hidden_dims: tuple = (64,64),
    kernel_size: int = 3,
    which: str = "best",
    device: str = None,
    save_arrays: bool = True,
    predictions_dir: str = None,
):
    """
    Run inference on the test set using a ConvLSTM model from train_radar_model.

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
    train_val_test_split : tuple, optional
        Tuple/list of three floats (train, val, test) that sum to 1.0 (default: (0.7, 0.15, 0.15)).
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
        Whether to save predictions and targets as .npy files (default: True).
    predictions_dir : str, optional
        Directory to save large prediction/target files (default: same as run_dir).
        If None, files are saved in run_dir. If specified, creates the directory if it doesn't exist.

    Returns
    -------
    pred_all : np.ndarray
        Array of shape (N, C, H, W) containing predicted radar reflectivity values.
    tgt_all : np.ndarray
        Array of shape (N, C, H, W) containing ground truth radar reflectivity values.
        MSE metrics are saved in run_dir/results/ as JSON file.
    """

    device = device or "cpu"
    run_dir = Path(run_dir)
    ckpt    = run_dir / ("best_val.pt" if which=="best" else "latest.pt")

    # Determine where to save predictions
    if predictions_dir is None:
        predictions_dir = run_dir
    else:
        predictions_dir = Path(predictions_dir)
        predictions_dir.mkdir(parents=True, exist_ok=True)

    cube = np.load(npy_path); cube[cube<0]=0

    T, C, H, W = cube.shape
    if not (isinstance(train_val_test_split, (tuple, list)) and len(train_val_test_split) == 3):
        raise ValueError("train_val_test_split must be a tuple/list of three floats (train, val, test)")
    if not abs(sum(train_val_test_split) - 1.0) < 1e-6:
        raise ValueError(f"train_val_test_split must sum to 1.0, got {train_val_test_split} (sum={sum(train_val_test_split)})")
    train_frac, val_frac, test_frac = train_val_test_split
    n_total = T - seq_len_in - seq_len_out + 1
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)
    idx_test = list(range(n_train + n_val, n_total))
    ds      = RadarWindowDataset(cube, seq_len_in, seq_len_out)
    test_ds  = Subset(ds, idx_test)
    dl      = DataLoader(test_ds, batch_size, shuffle=False)

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
            out = model(xb).cpu().numpy()
            preds.append(out)
            gts.append(yb.numpy())

    pred_all = np.concatenate(preds,axis=0)
    tgt_all  = np.concatenate(gts,axis=0)
    
    # Compute MSE for different reflectivity ranges
    ranges = [(0, 20), (20, 35), (35, 45), (45, 100)]
    mse_by_range = compute_mse_by_ranges(pred_all, tgt_all, ranges)
    
    # Save MSE metrics as JSON in results dir
    results_dir = run_dir / "results"
    results_dir.mkdir(exist_ok=True)
    import json
    with open(results_dir / "mse_by_ranges.json", "w") as f:
        json.dump(mse_by_range, f, indent=2)
    print("MSE by reflectivity range:")
    for range_name, mse in mse_by_range.items():
        print(f"{range_name}: {mse:.4f}")
    
    if save_arrays:
        np.save(predictions_dir/"test_preds_dBZ.npy",   pred_all)
        np.save(predictions_dir/"test_targets_dBZ.npy", tgt_all)
        print(f"Saved test_preds_dBZ.npy + test_targets_dBZ.npy → {predictions_dir}")
        print(f"Saved mse_by_ranges.json → {results_dir}")

    return pred_all, tgt_all


# CLI helper 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ConvLSTM radar forecasting model without normalization")
    subparsers = parser.add_subparsers(dest="command")

    # Subparser for training
    train_parser = subparsers.add_parser("train", help="Train ConvLSTM radar forecasting model")
    train_parser.add_argument("--save_dir", type=str, required=True, help="Directory to save model checkpoints and stats")
    train_parser.add_argument("--hidden_dims", type=str, required=True, help="Hidden dimensions as tuple, e.g., (64, 64)")
    train_parser.add_argument("--kernel_size", type=int, required=True, help="Kernel size (must be odd number)")

    # Optional arguments
    train_parser.add_argument("--npy_path", type=str, default="data/processed/ZH_radar_dataset.npy", help="Path to input .npy radar file")
    train_parser.add_argument("--seq_len_in", type=int, default=10, help="Input sequence length (default: 10)")
    train_parser.add_argument("--seq_len_out", type=int, default=1, help="Output sequence length (default: 1)")
    train_parser.add_argument("--train_val_test_split", type=str, default="(0.7,0.15,0.15)", help="Tuple/list of three floats (train, val, test) that sum to 1.0, e.g., (0.7,0.15,0.15)")
    train_parser.add_argument("--batch_size", type=int, default=1, help="Batch size (default: 4)")
    train_parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    train_parser.add_argument("--epochs", type=int, default=15, help="Number of epochs (default: 15)")
    train_parser.add_argument("--device", type=str, default='cuda', help="Device to train on ('cuda' or 'cpu')")
    train_parser.add_argument("--loss_name", type=str, default="mse", help="Loss function: mse or weighted_mse")
    train_parser.add_argument("--loss_weight_thresh", type=float, default=40.0,
                    help="Threshold in dBZ to apply higher loss weighting (default: 40.0)")
    train_parser.add_argument("--loss_weight_high", type=float, default=10.0,
                        help="Weight multiplier for pixels above threshold (default: 10.0)")
    train_parser.add_argument("--wandb_project", type=str, default="radar-forecasting", help="wandb project name")
    train_parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    train_parser.add_argument("--early_stopping_patience", type=int, default=10, help="Number of epochs with no improvement before early stopping (default: 10). Set to 0 or negative to disable early stopping.")

    # Subparser for test
    test_parser = subparsers.add_parser("test", help="Run test and compute MSE by reflectivity range")
    test_parser.add_argument("--npy_path", type=str, required=True, help="Path to input .npy radar file")
    test_parser.add_argument("--run_dir", type=str, required=True, help="Directory containing model checkpoints and stats")
    test_parser.add_argument("--seq_len_in", type=int, default=10, help="Input sequence length (default: 10)")
    test_parser.add_argument("--seq_len_out", type=int, default=1, help="Output sequence length (default: 1)")
    test_parser.add_argument("--train_val_test_split", type=str, default="(0.7,0.15,0.15)", help="Tuple/list of three floats (train, val, test) that sum to 1.0, e.g., (0.7,0.15,0.15)")
    test_parser.add_argument("--batch_size", type=int, default=4, help="Batch size (default: 4)")
    test_parser.add_argument("--hidden_dims", type=str, default="(64,64)", help="Hidden dimensions as tuple, e.g., (64, 64)")
    test_parser.add_argument("--kernel_size", type=int, default=3, help="Kernel size (default: 3)")
    test_parser.add_argument("--which", type=str, default="best", help="Which checkpoint to load: 'best' or 'latest'")
    test_parser.add_argument("--device", type=str, default=None, help="Device to run inference on (default: 'cpu')")
    test_parser.add_argument("--save_arrays", type=bool, default=True, help="Whether to save predictions and targets as .npy files")
    test_parser.add_argument("--predictions_dir", type=str, default=None, help="Directory to save large prediction/target files (default: same as run_dir)")

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

    train_val_test_split = ast.literal_eval(args.train_val_test_split)
    if args.command == "train":
        # Save arguments to save_dir/train_args.json
        import json, os
        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, "train_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        train_radar_model(
            npy_path=args.npy_path,
            save_dir=args.save_dir,
            args=args,
            seq_len_in=args.seq_len_in,
            seq_len_out=args.seq_len_out,
            train_val_test_split=train_val_test_split,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden_dims=hidden_dims,
            kernel_size=args.kernel_size,
            epochs=args.epochs,
            device=args.device,
            loss_name=args.loss_name,
            loss_weight_thresh=args.loss_weight_thresh,
            loss_weight_high=args.loss_weight_high,
            wandb_project=args.wandb_project,
            early_stopping_patience=args.early_stopping_patience,
        )
    elif args.command == "test":
        # Save arguments to run_dir/test_args.json
        import json, os
        os.makedirs(args.run_dir, exist_ok=True)
        with open(os.path.join(args.run_dir, "test_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        predict_test_set(
            npy_path=args.npy_path,
            run_dir=args.run_dir,
            seq_len_in=args.seq_len_in,
            seq_len_out=args.seq_len_out,
            train_val_test_split=train_val_test_split,
            batch_size=args.batch_size,
            hidden_dims=hidden_dims,
            kernel_size=args.kernel_size,
            which=args.which,
            device=args.device,
            save_arrays=args.save_arrays,
            predictions_dir=args.predictions_dir,
        )