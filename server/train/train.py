#!/usr/bin/env python3
"""
连续手势分类：密集时间头，~50ms 步长。严格按论文管线：
双通道 2x60x96 回声轮廓（原始+差分）-> 窗口内归一化（论文 3.3.2）
-> 帧 CNN（只下采样距离轴）-> 因果 Transformer -> 密集分类头。
损失：12 个输出步上的普通交叉熵，无 mask/类权重/label smoothing。

每次运行自动从 last.pt 续训，每个 epoch 覆盖保存 last.pt。

Usage (project python 3.10):
  python3 train.py [--epochs 60] [--batch 32] [--last last.pt] [--reset]
"""

import argparse
import contextlib
import os
import signal
import time

import numpy as np
import torch

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # timm 预训练权重走镜像

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

OUT_START, OUT_STRIDE = 48, 4                      # 后半窗口、~54ms 一步
OUT_IDX = list(range(OUT_START, 96, OUT_STRIDE))   # 12 个密集输出步
D_MODEL = 256


@contextlib.contextmanager
def no_interrupt():
    """保存期间屏蔽 Ctrl+C：挂起时按下的 SIGINT 被记账，
    保存完成后补发 KeyboardInterrupt，保证退出不丢失、文件不损坏。"""
    if hasattr(signal, 'SIGINT') and signal.getsignal(signal.SIGINT) is signal.default_int_handler:
        pending = []

        def handler(signum, frame):
            pending.append(True)

        prev = signal.signal(signal.SIGINT, handler)
        try:
            yield
        finally:
            signal.signal(signal.SIGINT, prev)
            if pending:
                print('\nCtrl+C 在保存期间被拦截，保存已完成，现在退出')
                raise KeyboardInterrupt
    else:
        yield


def safe_save(ck, path):
    """原子保存 + 全链路容错：先写 tmp 再 rename。
    写盘失败（杀毒/索引器抢文件、I/O 抖动）时删掉残 tmp 重试；
    rename 被占用时重试；持续失败则落盘为备份名，绝不让保存失败打断训练。"""
    tmp = path + '.tmp'
    for attempt in range(3):                        # 整个 写+rename 最多试 3 次
        try:
            torch.save(ck, tmp)
        except (OSError, RuntimeError):             # torch 写 zip 失败抛 RuntimeError
            if attempt == 2:
                raise
            print(f'警告: 检查点写入失败，{attempt + 2}/3 重试')
            try:
                os.remove(tmp)                      # 清理残缺 tmp 再重来
            except OSError:
                pass
            time.sleep(2.0)
            continue
        for _ in range(5):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                time.sleep(1.0)
    backup = f'{path}.ep{ck["epoch"]}'
    os.replace(tmp, backup)
    print(f'警告: {path} 被占用，本 epoch 检查点已存为 {backup}')


# ------------------------------------------------------------ dataset

class DenseDataset(Dataset):
    """2x60x96 窗口 -> 距离轴 pad 到 64，窗口内逐通道归一化（论文 3.3.2）。"""

    def __init__(self, X, Y, augment=False, drop=0.0, win_label=False, rng=None):
        self.X = np.pad(X, ((0, 0), (0, 0), (0, 4), (0, 0)), mode='edge')
        self.Y = Y
        # win_label=True 时返回窗口级单标签（后半窗多数表决，numpy 算好，MPS 无 mode 算子）
        self.win_label = win_label
        self.augment = augment
        self.drop = drop
        self.rng = rng or np.random.default_rng()

    def __len__(self):
        return len(self.Y)

    @staticmethod
    def _shift_zero(w, s):
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
            # 距离轴小位移（残差漂移）+ 幅度抖动
            x = self._shift_zero(x, int(r.integers(-2, 3)))
            if r.random() < 0.8:
                x *= r.uniform(0.95, 1.05)
            # drop 机制：随机抹掉时间/距离段（SpecAugment 式），
            # 迫使模型不依赖局部纹理性过拟合
            if self.drop > 0 and r.random() < self.drop:
                t0 = int(r.integers(0, 96 - 15))
                x[:, :, t0:t0 + int(r.integers(5, 16))] = 0
            if self.drop > 0 and r.random() < self.drop:
                d0 = int(r.integers(0, 64 - 10))
                x[:, d0:d0 + int(r.integers(5, 11)), :] = 0
        # 窗口内归一化（论文 3.3.2）
        for c in range(x.shape[0]):
            mu, sd = x[c].mean(), x[c].std()
            x[c] = (x[c] - mu) / (sd + 1e-6)
        if self.win_label:
            lab = int(np.bincount(self.Y[i][48:]).argmax())
            return (torch.from_numpy(x.astype(np.float32)),
                    torch.tensor(lab, dtype=torch.long))
        return (torch.from_numpy(x.astype(np.float32)),
                torch.tensor(self.Y[i], dtype=torch.long))


# ------------------------------------------------------------ model

class DenseGestureModel(nn.Module):
    """帧 CNN（时间步长 1）+ 因果 Transformer + 密集分类头。"""

    def __init__(self, num_classes, d_model=D_MODEL, layers=4, dropout=0.2):
        super().__init__()
        self.frame_cnn = nn.Sequential(
            nn.Conv2d(2, 64, 3, stride=(2, 1), padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=(2, 1), padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, d_model, 3, stride=(2, 1), padding=1), nn.BatchNorm2d(d_model), nn.ReLU(),
            nn.AvgPool2d(kernel_size=(8, 1)),  # 距离 8 -> 1，时间保持 96
        )
        self.pos = nn.Parameter(0.02 * torch.randn(96, d_model))
        layer = nn.TransformerEncoderLayer(d_model, nhead=8, dim_feedforward=4 * d_model,
                                           dropout=dropout, batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, num_classes))
        # register_buffer 随 model.to(device) 迁移，避免每次 forward 做 .to()；
        # persistent=False 不进 state_dict，保证与旧 last.pt 续训兼容
        self.register_buffer('_causal', torch.full((96, 96), float('-inf')).triu(1),
                             persistent=False)

    def forward(self, x):
        f = self.frame_cnn(x).squeeze(2)                 # (B, D, 96)
        f = f.permute(0, 2, 1) + self.pos                # (B, 96, D)
        h = self.temporal(f, mask=self._causal)
        return self.head(h[:, OUT_IDX])                  # (B, 12, K)


class _CausalDilatedBlock(nn.Module):
    """因果空洞卷积残差块：左侧补 2*dilation 保证只看过去。"""

    def __init__(self, d, dil, dropout):
        super().__init__()
        self.pad = 2 * dil
        self.conv = nn.Conv1d(d, d, 3, dilation=dil)
        self.bn = nn.BatchNorm1d(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = F.pad(x, (self.pad, 0))
        h = self.conv(h)
        return x + self.drop(F.relu(self.bn(h)))


class CnnDenseModel(nn.Module):
    """帧 CNN + 因果空洞 1D 卷积时间栈 + 密集头。
    用空洞卷积替代 Transformer 的 96² 注意力，MPS 上快数倍；
    空洞循环 (1,2,4,8) 堆叠感受野，layers=8 时覆盖 ~61 帧历史。"""

    def __init__(self, num_classes, d_model=D_MODEL, layers=8, dropout=0.2):
        super().__init__()
        self.frame_cnn = nn.Sequential(
            nn.Conv2d(2, 64, 3, stride=(2, 1), padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=(2, 1), padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, d_model, 3, stride=(2, 1), padding=1), nn.BatchNorm2d(d_model), nn.ReLU(),
            nn.AvgPool2d(kernel_size=(8, 1)),
        )
        self.temporal = nn.Sequential(*[
            _CausalDilatedBlock(d_model, 2 ** (i % 4), dropout) for i in range(layers)
        ])
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, num_classes))

    def forward(self, x):
        f = self.frame_cnn(x).squeeze(2)                 # (B, D, 96)
        h = self.temporal(f)                             # (B, D, 96) 因果
        return self.head(h.permute(0, 2, 1)[:, OUT_IDX])  # (B, 12, K)


FASTVIT_LOCAL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'fastvit_t12_in1k.safetensors')


class FastViTModel(nn.Module):
    """论文同款主干：FastViT-T12（ImageNet 预训练，卷积为主快而稳）。
    整窗 2×64×96 当时空图像一次前向，全局池化 → 10 类头，窗口级单标签。"""

    def __init__(self, num_classes, dropout=0.2, pretrained=True):
        super().__init__()
        import timm
        self.backbone = timm.create_model('fastvit_t12', pretrained=False,
                                          in_chans=2, num_classes=0)
        if pretrained and not os.path.exists(FASTVIT_LOCAL):
            # 本地无权重时回退在线下载（需能访问 huggingface，可设 HTTPS_PROXY/HF_ENDPOINT）
            print('本地无预训练权重，尝试在线下载...'
                  '（可先 python download_fastvit_weights.py 预先下载）')
            self.backbone = timm.create_model('fastvit_t12', pretrained=True,
                                              in_chans=2, num_classes=0)
        elif pretrained:
            # 本地加载：与 timm 在线路径等价，绕过 huggingface_hub 的代理兼容问题
            from safetensors.torch import load_file
            from timm.models._builder import adapt_input_conv
            sd = load_file(FASTVIT_LOCAL)
            # 3→2 输入通道适配：timm 在线路径只适配首卷积，遇 conv_scale 等
            # 其他按输入通道建的卷积会直接崩，这里对所有输入维不匹配的卷积逐个适配
            model_sd = self.backbone.state_dict()
            adapted, dropped = [], []
            for k, v in list(sd.items()):
                target = model_sd.get(k)
                if target is None or target.shape == v.shape:
                    continue
                if v.dim() == 4 and target.shape[0] == v.shape[0] and target.shape[2:] == v.shape[2:]:
                    try:
                        sd[k] = adapt_input_conv(target.shape[1], v)
                        adapted.append(k)
                        continue
                    except NotImplementedError:
                        pass
                del sd[k]                        # 形状无法适配 → 丢弃，随机初始化
                dropped.append(k)
            sd = {k: v for k, v in sd.items() if not k.startswith('head')}
            msg = self.backbone.load_state_dict(sd, strict=False)
            unexpected_missing = [k for k in msg.missing_keys if k not in dropped]
            if unexpected_missing:
                raise RuntimeError(f'fastvit 本地权重缺键: {unexpected_missing[:5]}')
            print(f'fastvit ImageNet 预训练权重已加载（本地 {os.path.basename(FASTVIT_LOCAL)}）'
                  f' 通道适配 {adapted}，随机初始化 {dropped}')
        d = self.backbone.num_features
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, num_classes))

    def forward(self, x):
        return self.head(self.backbone(x))                 # (B, K)


def build_model(arch, num_classes, d_model, layers, dropout):
    if arch == 'cnn':
        return CnnDenseModel(num_classes, d_model=d_model, layers=layers, dropout=dropout)
    if arch == 'fastvit':
        return FastViTModel(num_classes, dropout=dropout)
    return DenseGestureModel(num_classes, d_model=d_model, layers=layers, dropout=dropout)


# ------------------------------------------------------------ training

def run_epoch(model, loader, device, criterion, opt=None, amp=False, scaler=None, dense=True):
    train_mode = opt is not None
    model.train(train_mode)
    loss_sum, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train_mode):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            # amp 时前向+损失在 fp16 autocast 下算，反向由 scaler 防下溢
            with torch.amp.autocast(device.type, enabled=amp):
                logits = model(x)
                if dense:
                    ys = y[:, OUT_IDX]                       # (B,12) 密集步
                    loss = criterion(logits.reshape(-1, logits.shape[-1]), ys.reshape(-1))
                else:
                    ys = y                                     # (B,) 数据集已给窗口标签
                    loss = criterion(logits, ys)
            if train_mode:
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            n = ys.numel()
            loss_sum += loss.item() * n
            with torch.no_grad():
                correct += (logits.argmax(-1) == ys).sum().item()
                total += n
    return loss_sum / max(total, 1), correct / max(total, 1)


def window_acc(model, loader, device, dense=True):
    """窗口级准确率：dense=12 步多数表决；单标签=argmax vs 后半窗多数标签。"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(-1)
            if dense:
                for p, t in zip(pred, y[:, OUT_IDX]):
                    pv, tv = p.cpu().numpy(), t.cpu().numpy()
                    correct += np.bincount(pv).argmax() == np.bincount(tv).argmax()
                    total += 1
            else:
                tw = y                                         # (B,) 窗口标签
                correct += (pred == tw).sum().item()
                total += len(tw)
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='dataset.npz')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--classes', type=int, default=10,
                    help='输出分类数固定 10（当前数据不足 10 类时空类留位，后续加手势不用改结构）')
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--workers', type=int, default=0,
                    help='DataLoader 并行预处理进程数（cuda 下建议 2-4）')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--lr-mult', type=float, default=1.0,
                    help='学习率倍率：所有 lr（含判别式 lr）乘以此值，10 即加速 10 倍；'
                         '续训时按与检查点内旧倍率的比例自动缩放')
    ap.add_argument('--wd', type=float, default=1e-3)
    ap.add_argument('--layers', type=int, default=4, help='transformer 层数 (2=small)；cnn 架构下为空洞块数（推荐 8）')
    ap.add_argument('--arch', default='fastvit', choices=['fastvit', 'transformer', 'cnn'],
                    help='模型结构：fastvit（默认，论文同款窗口单标签）/ transformer（密集头）/ cnn（空洞卷积）')
    ap.add_argument('--freeze', type=int, default=0,
                    help='fastvit 前 N epoch 冻结 backbone 只训头（0=不冻，直接判别式 lr 全参）')
    ap.add_argument('--dmodel', type=int, default=D_MODEL)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--drop', type=float, default=0.3,
                    help='训练时随机 drop 时间/距离段的概率（0 关闭）')
    ap.add_argument('--last', default='last.pt', help='每个 epoch 覆盖保存的续训检查点')
    ap.add_argument('--amp', action='store_true',
                    help='fp16 混合精度训练（仅 cuda 生效，Ampere 卡可再提速约 30-60%%）')
    ap.add_argument('--reset', action='store_true', help='忽略 last.pt 从头训练')
    ap.add_argument('--device', default=None, help='mps/cpu; auto-detect if omitted')
    ap.add_argument('--exclude-session', type=int, nargs='*', default=[],
                    help='丢弃指定 session 索引的窗口（如质量差的手动采集会话）')
    args = ap.parse_args()

    data = np.load(args.data)
    X, Y, session = data['X'], data['Y'], data['session']
    split = data['split'] if 'split' in data.files else None
    # 输出头固定 10 类（用户决策）；数据类数超过时自动扩展
    num_classes = max(args.classes, int(Y.max()) + 1)
    if args.exclude_session:
        keep = np.ones(len(Y), dtype=bool)
        for s in args.exclude_session:
            keep &= session != s
        X, Y, session = X[keep], Y[keep], session[keep]
        if split is not None:
            split = split[keep]
        print(f'excluded sessions {args.exclude_session}: {keep.sum()} windows kept')

    # 显式 train/test 目录划分 > 跨会话 > 随机
    if split is not None and len(np.unique(split)) > 1:
        val_mask = split == 'test'
    else:
        sessions = np.unique(session)
        if len(sessions) > 1:
            val_mask = session == sessions[-1]
        else:
            val_mask = np.random.default_rng(0).random(len(Y)) < 0.2
    tr_mask = ~val_mask
    dense = args.arch != 'fastvit'
    print(f'train {tr_mask.sum()} / val {val_mask.sum()} windows, K={num_classes}, arch={args.arch}')

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    if device.type == 'cuda':
        # Ampere+ 显卡用 TF32 加速矩阵乘，对这种小模型精度影响可忽略
        torch.set_float32_matmul_precision('high')
    elif args.amp:
        print('警告: --amp 仅 cuda 生效，当前设备忽略')
    amp = args.amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    print('device:', device, '| amp:', amp)

    # num_workers>0 时增强/归一化在子进程并行做，pin_memory 加速 H2D 拷贝
    pin = device.type == 'cuda'
    common_dl_kwargs = dict(num_workers=args.workers, pin_memory=pin,
                            persistent_workers=args.workers > 0)
    train_dl = DataLoader(DenseDataset(X[tr_mask], Y[tr_mask], augment=True, drop=args.drop,
                                       win_label=not dense),
                          batch_size=args.batch, shuffle=True, **common_dl_kwargs)
    val_dl = DataLoader(DenseDataset(X[val_mask], Y[val_mask], win_label=not dense),
                        batch_size=args.batch, **common_dl_kwargs)

    model = build_model(args.arch, num_classes, d_model=args.dmodel, layers=args.layers,
                        dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'params: {n_params:.2f}M')

    criterion = nn.CrossEntropyLoss()

    # 续训：从 last.pt 恢复模型/优化器/调度器/epoch 进度
    start_ep = 0
    ck = None
    if os.path.exists(args.last) and not args.reset:
        ck = torch.load(args.last, map_location='cpu', weights_only=True)
        # 只拦截影响权重形状的配置（arch/dmodel/layers/num_classes；fastvit 无视后两者）；
        # dropout/drop 不改变形状，改了可无缝续训
        if ck.get('arch', 'transformer') != args.arch \
                or (args.arch != 'fastvit' and (ck.get('dmodel', D_MODEL) != args.dmodel
                                                or ck.get('layers', 4) != args.layers)) \
                or int(ck['num_classes']) != num_classes:
            raise SystemExit(f'{args.last} 模型配置与当前参数不一致，如需重头训练请加 --reset')
        model.load_state_dict(ck['model'])
        start_ep = int(ck['epoch'])
        print(f'续训: 从 epoch {start_ep} 继续 ({args.last})')

    # fastvit：判别式学习率（backbone 预训练少动）；--freeze 仅对新训生效
    lr = args.lr * args.lr_mult
    frozen = args.freeze > 0 and args.arch == 'fastvit' and start_ep == 0
    if frozen:
        for p in model.backbone.parameters():
            p.requires_grad = False
        opt = torch.optim.AdamW(model.head.parameters(), lr=lr, weight_decay=args.wd)
    elif args.arch == 'fastvit':
        opt = torch.optim.AdamW([
            {'params': model.head.parameters(), 'lr': lr},
            {'params': model.backbone.parameters(), 'lr': lr * 0.1},
        ], weight_decay=args.wd)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    if ck is not None and start_ep > 0:
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        if sched.T_max != args.epochs:
            # 总 epoch 数变了：按新长度重排余弦曲线，否则旧 T_max 会让
            # lr 以旧周期来回振荡（CosineAnnealingLR 超出 T_max 后不收敛到 0）
            print(f'注意: --epochs 从 {sched.T_max} 改为 {args.epochs}，'
                  f'余弦退火已按新长度重排（当前 lr 从接近峰值平滑衰减）')
            sched.T_max = args.epochs
        # 期望首组 lr（head/单组）vs 检查点保存的初始 lr：
        # 统一覆盖 --lr 与 --lr-mult 两者的变化，同参重启 ratio=1 不会累计放大
        ratio = (args.lr * args.lr_mult) / ck['opt']['param_groups'][0]['initial_lr']
        if abs(ratio - 1.0) > 1e-9:
            # 等比缩放恢复出的当前 lr、初始 lr 和调度器基线，
            # 余弦曲线形状不变，只是幅度按新配置拉伸
            for g in opt.param_groups:
                g['lr'] *= ratio
                g['initial_lr'] *= ratio
            sched.base_lrs = [b * ratio for b in sched.base_lrs]
            print(f'注意: 学习率配置变化，已等比缩放 {ratio:.2f}x'
                  f'（lr {ck["opt"]["param_groups"][0]["initial_lr"]:.2e} -> '
                  f'{args.lr * args.lr_mult:.2e}）')

    for ep in range(start_ep, args.epochs):
        if frozen and ep == args.freeze:
            # 解冻 backbone，换判别式 lr 优化器继续
            for p in model.backbone.parameters():
                p.requires_grad = True
            opt = torch.optim.AdamW([
                {'params': model.head.parameters(), 'lr': lr},
                {'params': model.backbone.parameters(), 'lr': lr * 0.1},
            ], weight_decay=args.wd)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs - ep)
            frozen = False
        tl, ta = run_epoch(model, train_dl, device, criterion, opt, amp=amp, scaler=scaler,
                           dense=dense)
        vl, va = run_epoch(model, val_dl, device, criterion, amp=amp, scaler=scaler,
                           dense=dense)
        sched.step()
        print(f'[ep{ep:03d}] train {tl:.3f}/{ta:.2f} | val {vl:.3f}/{va:.2f}')
        # 每个 epoch 覆盖保存 last，随时可中断后续训
        # Ctrl+C 与保存互斥 + 原子写盘：中断要么发生在保存前，要么在保存后
        with no_interrupt():
            ck = {'model': model.state_dict(), 'opt': opt.state_dict(),
                  'sched': sched.state_dict(), 'epoch': ep + 1,
                  'num_classes': num_classes, 'layers': args.layers,
                  'dmodel': args.dmodel, 'dropout': args.dropout,
                  'arch': args.arch, 'val_step_acc': va,
                  'lr_mult': args.lr_mult}
            safe_save(ck, args.last)
    print(f'done -> {args.last} | window-acc {window_acc(model, val_dl, device, dense=dense):.2f}')


if __name__ == '__main__':
    main()
