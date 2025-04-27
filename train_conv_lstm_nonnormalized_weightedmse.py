"""
train_conv_lstm.py
------------------
Train a ConvLSTM on 4-D radar data with
 • z-score normalisation (stats taken ONLY from the training horizon)
 • chronological train/val split
 • weighted MSE that emphasises >30 dBZ echoes
 • automatic checkpoints (latest + best)
 • Weights & Biases integration for experiment tracking

Author: you
"""

from pathlib import Path
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import wandb  # Add wandb import

# ------------------------------------------------------------------
# 1)  Dataset – keep raw dBZ
# ------------------------------------------------------------------
class RadarWindowDataset(Dataset):
    def __init__(self, cube, seq_in, seq_out):
        X, Y = [], []
        last_start = cube.shape[0] - seq_in - seq_out + 1
        for t in range(last_start):
            X.append(cube[t       : t+seq_in   ])      # raw dBZ
            Y.append(cube[t+seq_in: t+seq_in+1].squeeze(0))
        self.X = np.stack(X).astype(np.float32)        # (N,S,C,H,W)
        self.Y = np.stack(Y).astype(np.float32)        # (N,C,H,W)

    def __len__(self):  return len(self.X)
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.Y[i])

# ------------------------------------------------------------------
# 2)  ConvLSTM blocks
# ------------------------------------------------------------------
class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid_ch, k=3):
        super().__init__()
        p = k // 2
        self.hid_ch = hid_ch
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, k, padding=p)

    def forward(self, x, h, c):
        i, f, o, g = torch.chunk(self.conv(torch.cat([x, h], 1)), 4, 1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c

    def init_hidden(self, B, H, W, dev):
        h = torch.zeros(B, self.hid_ch, H, W, device=dev)
        c = torch.zeros_like(h)
        return h, c

class ConvLSTM(nn.Module):
    def __init__(self, in_ch, hidden, k=3):
        super().__init__()
        self.layers = nn.ModuleList()
        for i, h in enumerate(hidden):
            self.layers.append(
                ConvLSTMCell(in_ch if i == 0 else hidden[i-1], h, k))
        self.to_out = nn.Conv2d(hidden[-1], in_ch, 1)

    def forward(self, x):             # x (B,S,C,H,W)
        B, S, _, H, W = x.shape
        dev = x.device
        h_list, c_list = [], []
        for cell in self.layers:
            h,c = cell.init_hidden(B,H,W,dev)
            h_list.append(h); c_list.append(c)

        for t in range(S):
            x_t = x[:, t]
            for i, cell in enumerate(self.layers):
                h_list[i], c_list[i] = cell(x_t, h_list[i], c_list[i])
                x_t = h_list[i]

        return self.to_out(x_t)       # (B,C,H,W)

# ------------------------------------------------------------------
# 3)  Training function
# ------------------------------------------------------------------
def train_radar_model(
        npy_path:   str,
        save_dir:   str,
        *,
        seq_len_in = 10,
        seq_len_out= 1,
        train_frac = 0.8,
        batch      = 4,
        lr         = 2e-4,
        hidden     = (64,64),
        kernel     = 3,
        hi_thresh  = 30.0,
        hi_weight  = 8.0,
        epochs     = 15,
        device     = "cuda" if torch.cuda.is_available() else "cpu"):

    # ---------- ensure output dir exists ----------
    save_dir = Path(save_dir)                 # convert once
    save_dir.mkdir(parents=True, exist_ok=True)
    # ---------- load cube ----------
    cube = np.load(npy_path)
    cube[cube < 0] = 0
    T, C, H, W = cube.shape
    print(f"Loaded {npy_path} → {cube.shape}")

    # ---------- chronological split ----------
    n_total = T - seq_len_in - seq_len_out + 1
    n_train = int(n_total * train_frac)

    # build datasets
    full_ds  = RadarWindowDataset(cube, seq_len_in, seq_len_out)
    train_ds = Subset(full_ds, list(range(0, n_train)))
    val_ds   = Subset(full_ds, list(range(n_train, n_total)))

    train_dl = DataLoader(train_ds, batch, shuffle=False)
    val_dl   = DataLoader(val_ds,   batch, shuffle=False)
    print(f"Samples train={len(train_ds)}  val={len(val_ds)}")

    # ---------- model, loss, opt ----------
    model = ConvLSTM(in_ch=C, hidden=hidden, k=kernel).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

# ---------------AUTO-RESUME-------------------------
    latest   = Path(save_dir) / "latest.pt"
    best_wts = Path(save_dir) / "best_val.pt"
    best_val = float("inf")
    start_ep = 1

    if latest.exists():
        chk = torch.load(latest, map_location=device)
        model.load_state_dict(chk["model"])
        optim.load_state_dict(chk["optim"])
        best_val = chk["best_val"]
        start_ep = chk["epoch"] + 1
        print(f"✔ Resumed from epoch {chk['epoch']}  (best val={best_val:.4f})")
# --------------------------------------------------

    # Initialize wandb after model and optimizer setup
    run_id = save_dir.name  # use your folder name as a stable run ID
    wandb.init(
        project="radar-forecasting",
        name=run_id,            
        id=run_id,               # this ties W&B to the same run
        resume="allow",              # auto-resume same run
        config={
            "seq_len_in": seq_len_in,
            "seq_len_out": seq_len_out,
            "train_frac": train_frac,
            "batch_size": batch,
            "learning_rate": lr,
            "hidden_dims": hidden,
            "kernel_size": kernel,
            "hi_threshold": hi_thresh,
            "hi_weight": hi_weight,
            "epochs": epochs,
            "device": device
        }
    )
    wandb.watch(model)  # Track model gradients and parameters

    hi_thr = torch.tensor(hi_thresh, device=device)
    hi_w   = torch.tensor(hi_weight, device=device)

    def weighted_mse(pred, target):
        """Both pred/target are raw dBZ tensors"""
        w = torch.where(target >= hi_thr, hi_w, 1.0)
        return (w * (pred - target) ** 2).mean()

    # ---------- checkpoint setup ----------
    latest   = save_dir/"latest.pt"
    best_wts = save_dir/"best_val.pt"
    best_val = float("inf")

    # ---------- training loop ----------
    def run_epoch(loader, train=True):
        model.train() if train else model.eval()
        total = 0.
        with torch.set_grad_enabled(train):
            for x,y in loader:
                x,y = x.to(device), y.to(device)
                pred = model(x)
                loss = weighted_mse(pred, y)
                if train:
                    optim.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(),1.)
                    optim.step()
                total += loss.item()*x.size(0)
        return total / len(loader.dataset)

    for ep in range(start_ep, epochs + 1):
        tr = run_epoch(train_dl, True)
        vl = run_epoch(val_dl,   False)
        print(f"[{ep:02d}/{epochs}] train {tr:.4f} | val {vl:.4f}")

        # Log metrics to wandb
        wandb.log({
            "train_loss": tr,
            "val_loss": vl,
            "epoch": ep
        })

        torch.save({'epoch':ep,
                    'model':model.state_dict(),
                    'optim':optim.state_dict(),
                    'best_val':best_val},
                   latest)

        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), best_wts)
            print("  🏆 new best saved")
            # Log best validation loss
            wandb.log({"best_val_loss": best_val})

    print("Done. Checkpoints stored in", save_dir.resolve())
    wandb.finish()  # End the wandb run

# ------------------------------------------------------------------
# 3b) predict_validation_set  (simplified)
# ------------------------------------------------------------------
def predict_validation_set(
        npy_path : str,
        run_dir  : str,
        *,
        seq_len_in=10, seq_len_out=1,
        train_frac=0.8, batch=4,
        hidden=(64,64), kernel=3,
        which_ckpt="best", device="cpu",
        save_arrays=True):

    run_dir = Path(run_dir)
    ckpt_f  = run_dir / ("best_val.pt" if which_ckpt=="best" else "latest.pt")

    cube = np.load(npy_path).astype(np.float32)
    C    = cube.shape[1]

    # rebuild split
    T       = cube.shape[0]
    n_total = T - seq_len_in - seq_len_out + 1
    n_train = int(n_total * train_frac)

    full_ds = RadarWindowDataset(cube, seq_len_in, seq_len_out)
    val_ds  = Subset(full_ds, list(range(n_train, n_total)))
    val_dl  = DataLoader(val_ds, batch_size=batch, shuffle=False)

    # model
    model = ConvLSTM(in_ch=C, hidden=hidden, k=kernel).to(device)
    state = torch.load(ckpt_f, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state); model.eval()

    preds, gts = [], []
    with torch.no_grad():
        for xb, yb in val_dl:
            xb = xb.to(device)
            pred = model(xb).cpu()
            preds.append(pred)
            gts.append(yb)            # already on CPU
    pred_all = torch.cat(preds).numpy()
    tgt_all  = torch.cat(gts ).numpy()

    if save_arrays:
        np.save(run_dir/"val_preds_dBZ.npy",   pred_all)
        np.save(run_dir/"val_targets_dBZ.npy", tgt_all)
        print("Saved arrays in", run_dir)

    return pred_all, tgt_all

# # ------------------------------------------------------------------
# # 4)  CLI helper  (optional)
# # ------------------------------------------------------------------
# if __name__ == "__main__":
#     import argparse, textwrap
#     p = argparse.ArgumentParser(
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         description=textwrap.dedent("""\
#             Train ConvLSTM on radar cube.

#             Example:
#               python train_conv_lstm.py Data/ZH_radar_dataset.npy runs/run1
#         """))
#     p.add_argument("npy_path")
#     p.add_argument("save_dir")
#     p.add_argument("--epochs", type=int, default=15)
#     args = p.parse_args()

#     train_radar_model(args.npy_path, args.save_dir, epochs=args.epochs)