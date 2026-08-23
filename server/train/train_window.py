#!/usr/bin/env python3
"""
Gesture classification training: FastViT-T12 (paper backbone) + staged
fine-tuning on 2x64x96 echo profile windows.

Usage (project python 3.10):
  /Library/Frameworks/Python.framework/Versions/3.10/bin/python3 train.py \
      [--epochs1 5] [--epochs2 40] [--batch 32] [--ckpt checkpoint.pt]

Staged training:
  Phase 1: backbone frozen (ImageNet features), train head only.
  Phase 2: full fine-tune, discriminative lrs (backbone 10x smaller),
           early stopping on val loss.
Split: leave-last-session-out (cross-session, paper protocol); falls back
to trial-level split when only one session exists.
"""

import argparse
import os

import numpy as np
import torch

# HF is often unreachable from this network; mirror fallback for weight download
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
import torch.nn as nn
import timm
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------------------ dataset

class GestureDataset(Dataset):
    """2x60x96 windows -> padded 64x96, per-window normalized, augmented."""

    def __init__(self, X, y, augment=False, rng=None):
        # pad distance axis 60 -> 64 (multiple of backbone stride 32)
        self.X = np.pad(X, ((0, 0), (0, 0), (0, 4), (0, 0)), mode='edge')
        self.y = y
        self.augment = augment
        self.rng = rng or np.random.default_rng()

    def __len__(self):
        return len(self.y)

    @staticmethod
    def _shift_zero(w, s):
        """Shift along distance axis (axis=1), zero-fill."""
        out = np.zeros_like(w)
        n = w.shape[1]
        if s > 0:
            out[:, :n - s] = w[:, s:]
        elif s < 0:
            out[:, -s:] = w[:, :n + s]
        return out

    def __getitem__(self, i):
        x = self.X[i].copy()
        if self.augment:
            r = self.rng
            # vertical shift +/-8 px (paper uses +/-5; wider for our drift margin)
            x = self._shift_zero(x, int(r.integers(-2, 3)))
            # amplitude variation
            if r.random() < 0.8:
                x *= r.uniform(0.95, 1.05)
            # time / distance masking
            if r.random() < 0.2:
                t0 = r.integers(0, 96 - 15)
                x[:, :, t0:t0 + r.integers(5, 16)] = 0
            if r.random() < 0.2:
                d0 = r.integers(0, 64 - 10)
                x[:, d0:d0 + r.integers(5, 11), :] = 0
        # per-window normalization (paper 3.3.2)
        for c in range(x.shape[0]):
            mu, sd = x[c].mean(), x[c].std()
            x[c] = (x[c] - mu) / (sd + 1e-6)
        return torch.from_numpy(x.astype(np.float32)), torch.tensor(self.y[i], dtype=torch.long)


# ------------------------------------------------------------ training

def freeze_backbone(model, frozen=True):
    for name, p in model.named_parameters():
        p.requires_grad = (not frozen) or ('head' in name)


def run_epoch(model, loader, device, criterion, opt=None):
    train_mode = opt is not None
    model.train(train_mode)
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(train_mode):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if train_mode:
                opt.zero_grad()
                loss.backward()
                opt.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
    return loss_sum / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='dataset.npz')
    ap.add_argument('--epochs1', type=int, default=5)
    ap.add_argument('--epochs2', type=int, default=40)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--ckpt', default='checkpoint.pt')
    ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--exclude-session', type=int, nargs='*', default=[])
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    data = np.load(args.data)
    X, session = data['X'], data['session']
    if 'y' in data.files:
        y = data['y']
    else:  # 密集标签 -> 窗口级多数表决标签（输出区 48:）
        y = np.array([np.bincount(w[48:]).argmax() for w in data['Y']])
    split = data['split'] if 'split' in data.files else None
    num_classes = int(y.max()) + 1
    if args.exclude_session:
        keep = np.ones(len(y), dtype=bool)
        for s in args.exclude_session:
            keep &= session != s
        X, y, session = X[keep], y[keep], session[keep]
        if split is not None:
            split = split[keep]
        print(f'excluded sessions {args.exclude_session}: {keep.sum()} windows kept')

    # explicit train/test dirs > cross-session > random
    if split is not None and len(np.unique(split)) > 1:
        val_mask = split == 'test'
    else:
        sessions = np.unique(session)
        if len(sessions) > 1:
            val_mask = session == sessions[-1]
        else:
            rng = np.random.default_rng(0)
            val_mask = rng.random(len(y)) < 0.2
    tr_mask = ~val_mask
    print(f'train {tr_mask.sum()} windows / val {val_mask.sum()} windows, '
          f'classes {np.bincount(y).tolist()}')

    train_ds = GestureDataset(X[tr_mask], y[tr_mask], augment=True)
    val_ds = GestureDataset(X[val_mask], y[val_mask])
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print('device:', device)
    model = timm.create_model('fastvit_t12', pretrained=True, in_chans=2,
                              num_classes=num_classes, drop_rate=0.2).to(device)

    counts = np.bincount(y[tr_mask], minlength=num_classes).astype(np.float64)
    weight = torch.tensor(counts.sum() / (num_classes * counts + 1e-6),
                          dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    # ---- Phase 1: head only
    freeze_backbone(model, frozen=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=1e-3, weight_decay=1e-4)
    for ep in range(args.epochs1):
        tl, ta = run_epoch(model, train_dl, device, criterion, opt)
        vl, va = run_epoch(model, val_dl, device, criterion)
        print(f'[p1 ep{ep}] train loss {tl:.3f} acc {ta:.2f} | val loss {vl:.3f} acc {va:.2f}')

    # ---- Phase 2: full fine-tune, discriminative lrs, early stopping
    freeze_backbone(model, frozen=False)
    head_params = [p for n, p in model.named_parameters() if 'head' in n]
    body_params = [p for n, p in model.named_parameters() if 'head' not in n]
    opt = torch.optim.AdamW([
        {'params': body_params, 'lr': 1e-5},
        {'params': head_params, 'lr': 1e-4},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs2)

    best_val, bad, best_state = float('inf'), 0, None
    for ep in range(args.epochs2):
        tl, ta = run_epoch(model, train_dl, device, criterion, opt)
        vl, va = run_epoch(model, val_dl, device, criterion)
        sched.step()
        print(f'[p2 ep{ep}] train loss {tl:.3f} acc {ta:.2f} | val loss {vl:.3f} acc {va:.2f}')
        if vl < best_val - 1e-4:
            best_val, bad, best_state = vl, 0, {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f'early stop at ep{ep}')
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({'model': model.state_dict(), 'num_classes': num_classes}, args.ckpt)
    vl, va = run_epoch(model, val_dl, device, criterion)
    print(f'final val loss {vl:.3f} acc {va:.2f} -> {args.ckpt}')


if __name__ == '__main__':
    main()
