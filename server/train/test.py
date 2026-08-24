#!/usr/bin/env python3
"""
读取 last.pt 在指定 split 上评估（默认 test）。

输出：step-acc（密集步准确率）、window-acc（窗口多数表决准确率）、
混淆矩阵与每类 recall。

Usage (project python 3.10):
  python3 test.py [--split test/train/all] [--last last.pt] [--data dataset.npz] [--device cuda/mps/cpu]
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
    ap.add_argument('--batch', type=int, default=256, help='评估无梯度，batch 可开大')
    ap.add_argument('--workers', type=int, default=2, help='DataLoader 预处理进程数')
    ap.add_argument('--device', default=None, help='cuda/mps/cpu; auto-detect if omitted')
    ap.add_argument('--split', default='test', choices=['test', 'train', 'all'],
                    help='评估哪个 split')
    args = ap.parse_args()

    ck = torch.load(args.last, map_location='cpu', weights_only=True)
    num_classes = int(ck['num_classes'])
    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')
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
    test_dl = DataLoader(DenseDataset(X[mask], Y[mask]), batch_size=args.batch,
                         num_workers=args.workers, pin_memory=device.type == 'cuda',
                         persistent_workers=args.workers > 0)

    K = num_classes
    pred_list, targ_list = [], []
    with torch.no_grad():
        for x, y in test_dl:
            x = x.to(device, non_blocking=True)
            pred = model(x).argmax(-1)                     # (B, 12)
            pred_list.append(pred.cpu().numpy())
            targ_list.append(y[:, OUT_IDX].numpy())

    # 全量向量化统计，避免逐样本 Python 循环
    P = np.concatenate(pred_list).reshape(-1)
    T = np.concatenate(targ_list).reshape(-1)
    cm = np.bincount(T * K + P, minlength=K * K).reshape(K, K)
    Pw = np.concatenate(pred_list)                          # (N, 12)
    Tw = np.concatenate(targ_list)
    idx = np.arange(len(Pw))[:, None]
    hp = np.zeros((len(Pw), K), dtype=np.int64)
    ht = np.zeros((len(Pw), K), dtype=np.int64)
    np.add.at(hp, (idx, Pw), 1)
    np.add.at(ht, (idx, Tw), 1)
    w_correct = int((hp.argmax(1) == ht.argmax(1)).sum())
    w_total = len(Pw)

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
