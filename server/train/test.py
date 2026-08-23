#!/usr/bin/env python3
"""
读取 last.pt 在指定 split 上评估（默认 test）。

输出：step-acc（密集步准确率）、window-acc（窗口多数表决准确率）、
混淆矩阵与每类 recall。

Usage (project python 3.10):
  python3 test.py [--split test/train/all] [--last last.pt] [--data dataset.npz] [--device mps/cpu]
"""

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from train import DenseGestureModel, DenseDataset, OUT_IDX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--last', default='last.pt')
    ap.add_argument('--data', default='dataset.npz')
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--device', default=None, help='mps/cpu; auto-detect if omitted')
    ap.add_argument('--split', default='test', choices=['test', 'train', 'all'],
                    help='评估哪个 split')
    args = ap.parse_args()

    ck = torch.load(args.last, map_location='cpu', weights_only=True)
    num_classes = int(ck['num_classes'])
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = DenseGestureModel(num_classes, d_model=ck.get('dmodel', 256),
                              layers=ck.get('layers', 4),
                              dropout=ck.get('dropout', 0.2)).to(device)
    model.load_state_dict(ck['model'])
    model.eval()

    data = np.load(args.data)
    X, Y, split = data['X'], data['Y'], data['split']
    if args.split == 'train':
        mask = split == 'train'
    elif args.split == 'test':
        mask = split == 'test'
    else:
        mask = np.ones(len(split), dtype=bool)
    test_dl = DataLoader(DenseDataset(X[mask], Y[mask]), batch_size=args.batch)

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    w_correct, w_total = 0, 0
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(-1)                     # (B, 12)
            for p, t in zip(pred, y[:, OUT_IDX]):
                pv, tv = p.cpu().numpy(), t.cpu().numpy()
                for tt, pp in zip(tv, pv):
                    cm[tt, pp] += 1
                w_correct += np.bincount(pv).argmax() == np.bincount(tv).argmax()
                w_total += 1

    step_acc = cm.trace() / max(cm.sum(), 1)
    print(f'{args.last} (epoch {ck.get("epoch", "?")}, '
          f'val step-acc at save {ck.get("val_step_acc", float("nan")):.2f})')
    print(f'{args.split}: step-acc {step_acc:.3f} | window-acc {w_correct / max(w_total, 1):.3f} '
          f'({w_total} windows)')
    print('混淆矩阵 (行=真实, 列=预测):')
    print(cm)
    for k in range(num_classes):
        tot = cm[k].sum()
        print(f'  class {k}: recall {cm[k, k] / max(tot, 1):.2f} ({cm[k, k]}/{tot})')


if __name__ == '__main__':
    main()
