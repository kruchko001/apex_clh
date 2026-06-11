import os
from datetime import datetime

import numpy as np
import torch as th
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

import tron_paper_BH  # noqa: F401
from tron_paper.model.mrl_net import MRLActorCritic
from tron_paper_BH.collect import DEFAULT_DATA

DEFAULT_SAVE = "./tron_paper_BH_checkpoints"


def train_bc(
    data_path: str = DEFAULT_DATA,
    save_dir: str = DEFAULT_SAVE,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 3e-4,
    val_frac: float = 0.1,
    patience: int = 5,
    min_epochs: int = 3,
    device: str = None,
    verbose: int = 1,
):
    device = device or ("cuda" if th.cuda.is_available() else "cpu")
    data = np.load(data_path)
    obs = th.from_numpy(data["obs"])
    acts = th.from_numpy(data["acts"]).long()
    ds = TensorDataset(obs, acts)
    n_val = max(int(len(ds) * val_frac), 1)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=th.Generator().manual_seed(0))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = MRLActorCritic(stationary=True).to(device)
    opt = th.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_acc, n = 0.0, 0, 0
        for xb, ab in train_loader:
            xb, ab = xb.to(device), ab.to(device)
            logits, _ = model(xb)
            loss = F.cross_entropy(logits, ab)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(ab)
            train_acc += (logits.argmax(-1) == ab).sum().item()
            n += len(ab)

        model.eval()
        val_loss, val_acc, vn = 0.0, 0, 0
        with th.no_grad():
            for xb, ab in val_loader:
                xb, ab = xb.to(device), ab.to(device)
                logits, _ = model(xb)
                val_loss += F.cross_entropy(logits, ab).item() * len(ab)
                val_acc += (logits.argmax(-1) == ab).sum().item()
                vn += len(ab)

        tl, ta = train_loss / n, train_acc / n
        vl, va = val_loss / vn, val_acc / vn
        tag = ""
        if vl < best_val_loss:
            best_val_loss = vl
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
            tag = " *best*"
        elif epoch >= min_epochs and patience > 0:
            stale += 1

        if verbose:
            print(
                f"epoch {epoch}/{epochs}  "
                f"train_loss={tl:.4f} acc={ta:.3f}  "
                f"val_loss={vl:.4f} acc={va:.3f}{tag}"
            )

        if patience > 0 and epoch >= min_epochs and stale >= patience:
            if verbose:
                print(f"early stop at epoch {epoch} (best epoch {best_epoch}, val_loss={best_val_loss:.4f})")
            break

    os.makedirs(save_dir, exist_ok=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.cpu()
    pt_path = os.path.join(save_dir, "stationary_agent.pt")
    th.save(model.state_dict(), pt_path)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    extra = os.path.join(save_dir, f"stationary_bc_{ts}.pt")
    th.save(model.state_dict(), extra)
    if verbose:
        print(f"Saved {pt_path} (best epoch {best_epoch}, val_loss={best_val_loss:.4f})")
    return model, pt_path
