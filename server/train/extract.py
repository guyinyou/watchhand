#!/usr/bin/env python3
"""
Offline dataset extraction (dense version): raw PCM + label events ->
2x60x96 windows with per-frame labels and boundary loss masks.

Differences vs the old window-level version:
  - windows are sampled over the FULL timeline (rest/transitions included),
    so the dense head learns context and transition behavior
  - Y is a per-frame label vector (96,), enabling 50ms-step supervision
  - M is a loss mask (96,): steps within +/-BOUNDARY frames of a label change
    are excluded from the loss (acoustic evidence lags the key press)

Usage: python3 extract.py [out.npz]   (project python 3.10)
"""

import os
import sys
import glob

import numpy as np
from scipy.signal import lfilter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from watchhand_server import EchoProfileProcessor, BiquadFilter

WINDOW_FRAMES = 96     # 1.28 s model window (96 frames @ 75fps unified)
STRIDE_FRAMES = 22     # ~0.29 s window sampling stride
DIRECT_BIN = 5
REF_FS = 44100         # 统一距离网格参考采样率：所有新数据均为 44.1kHz（3.89mm/bin），
                       # 旧 48kHz 会话经插值对齐到该网格

# 轮廓处理算法版本：改动滤波/对齐/校准逻辑时手动 +1，使旧缓存自动失效
PROCESS_VERSION = 3
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'profile_cache')


def unify_distance_grid(profile, fs):
    """把距离轴插值到统一的物理距离网格（REF_FS 对应的 3.57mm/bin）。

    不同采样率的 bin 物理宽度不同（48kHz: 3.57mm, 44.1kHz: 3.89mm，差 ~9%），
    混合训练前必须对齐，否则同一手位置在不同设备上落在不同 bin。
    目标 bin i 对应距离 i*c/(2*REF_FS)，换算到源 bin 坐标为 i*fs/REF_FS。
    """
    if fs == REF_FS:
        return profile
    bins, n_frames = profile.shape
    tgt = np.arange(bins, dtype=np.float64) * (fs / REF_FS)
    i0 = np.clip(np.floor(tgt).astype(int), 0, bins - 1)
    i1 = np.clip(i0 + 1, 0, bins - 1)
    frac = (tgt - i0).astype(np.float32)[:, None]
    return (profile[i0] * (1.0 - frac) + profile[i1] * frac).astype(np.float32)


def load_or_compute_profile(base, samples, meta):
    """会话级缓存：滤波+互相关是耗时大头，按 (raw/meta 的 size+mtime, 算法版本)
    缓存轮廓；窗口切分/标签每次重做（便宜且随标签编辑更新）。
    不做设备校准：稳定伪影/漂移交给模型自己学（用户决策）。"""
    st = os.stat(base + '.raw')
    mt = os.stat(base + '.meta')
    key = f'v{PROCESS_VERSION}_{st.st_size}_{int(st.st_mtime)}_{int(mt.st_mtime)}'
    os.makedirs(CACHE_DIR, exist_ok=True)
    cpath = os.path.join(CACHE_DIR, os.path.basename(base) + '.npz')
    if os.path.exists(cpath):
        try:
            c = np.load(cpath)
            if str(c['key'].item()) == key:
                print(f'  cache hit: {os.path.basename(base)}')
                return c['P']
        except Exception:
            pass  # 缓存坏了就重算
    P = batch_profile(samples, meta)
    np.savez(cpath, P=P, key=np.array(key))
    return P


def load_session(base):
    meta = {}
    with open(base + '.meta') as f:
        for line in f:
            if '=' in line:
                k, v = line.strip().split('=', 1)
                meta[k] = v
    samples = np.fromfile(base + '.raw', dtype='<i2')
    events = []
    with open(base + '.labels') as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                events.append((int(parts[0]), int(parts[1])))
    return samples, meta, events


def _biquad_lfilter(x, coeffs):
    """用 scipy.signal.lfilter（C 实现）替代 Python 循环，快 10-50x。
    coeffs 格式 [b0, b1, b2, a1, a2]，与 watchhand_server 一致。"""
    b = np.array([coeffs[0], coeffs[1], coeffs[2]], dtype=np.float64)
    a = np.array([1.0, coeffs[3], coeffs[4]], dtype=np.float64)
    return lfilter(b, a, x).astype(np.float32)


def batch_profile(samples, meta):
    """Full-recording original echo profile (60 x nFrames), aligned.

    优化：
    1. 滤波用 scipy.signal.lfilter（C 实现，比 Python 循环快 10-50x）
    2. 相关用逐帧向量化点积（与 Android processChunk 数学等价），
       避免对整段录音做巨大 FFT（长录音时省内存且更快）

    对齐约定不变：lag = p_start + f*L + d，与 Android/服务端完全一致。
    """
    fs = int(meta['sample_rate'])
    L = int(meta['chirp_length'])
    bins = int(meta['distance_bins'])
    proc = EchoProfileProcessor(fs, float(meta['f_min']), float(meta['f_max']),
                                L, bins, int(meta['time_window_frames']))
    tx = proc.tx_chirp.astype(np.float64)

    # --- 1. 带通滤波：3 HP + 2 LP，scipy C 实现 ---
    x = samples.astype(np.float64)
    for _ in range(3):
        x = _biquad_lfilter(x, proc.hp_coeffs)
    for _ in range(2):
        x = _biquad_lfilter(x, proc.lp_coeffs)
    x = x.astype(np.float32)

    # --- 2. 对齐：用前 N_ALIGN 帧的相关找 p_start（与全段 FFT 结果一致） ---
    n_align = min(len(x) - L, L * 40)  # 前 40 个 chirp 周期足够确定相位
    corr_align = np.array([np.dot(x[k:k + L].astype(np.float64), tx)
                           for k in range(n_align)], dtype=np.float32)

    n0 = (len(corr_align) // L) * L
    b0 = int(np.argmax(np.abs(corr_align[:n0].reshape(-1, L)).mean(axis=0)))
    p_start = None
    for cand in [(b0 - DIRECT_BIN) % L, (DIRECT_BIN - b0) % L]:
        nf_c = (len(corr_align) - cand) // L - 2
        if nf_c <= 0:
            continue
        fr = corr_align[cand:cand + nf_c * L].reshape(nf_c, L)
        if 0 <= int(np.argmax(np.abs(fr[:, :bins]).mean(axis=0))) - DIRECT_BIN <= 2:
            p_start = cand
            break
    if p_start is None:
        p_start = (b0 - DIRECT_BIN) % L

    # --- 3. 逐帧向量化相关（与 Android processChunk 等价） ---
    # 每帧 bin d = dot(x[p_start + f*L + d : +L], tx)
    nf = (len(x) - p_start - bins + 1 - L + 1) // L
    if nf <= 0:
        nf = max((len(x) - p_start) // L - 2, 0)

    profile = np.zeros((bins, nf), dtype=np.float32)
    x64 = x.astype(np.float64)
    # sliding_window_view 零拷贝视图，按 stride=L 取帧（避免 Python 循环）
    windows = np.lib.stride_tricks.sliding_window_view(x64, L)  # (len-L+1, L)
    for d in range(bins):
        starts = p_start + np.arange(nf) * L + d
        valid = starts + L <= len(x)
        n_valid = int(valid.sum())
        if n_valid > 0:
            profile[d, :n_valid] = windows[starts[:n_valid]] @ tx

    return unify_distance_grid(profile, fs)


def frame_labels(events, n_frames, L):
    """Per-frame label vector from change events."""
    y = np.zeros(n_frames, dtype=np.int64)
    for off, lab in events:  # events sorted; sequential overwrite
        y[off // L:] = lab
    return y


def make_window(P, f0, wf):
    orig = P[:, f0:f0 + wf]
    diff = np.abs(orig)[:, 1:] - np.abs(orig)[:, :-1]
    diff = np.concatenate([diff, diff[:, -1:]], axis=1)
    return np.stack([orig, diff], axis=0).astype(np.float32)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('out', nargs='?', default='dataset.npz')
    ap.add_argument('--window', type=int, default=WINDOW_FRAMES,
                    help='窗口帧数（默认 96；如 32 帧短窗实验）')
    ap.add_argument('--stride', type=int, default=STRIDE_FRAMES,
                    help='滑窗步长（默认 22）')
    args = ap.parse_args()
    out_path = args.out
    wf, sf = args.window, args.stride
    print(f'窗口 {wf} 帧 / 步长 {sf} 帧')
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'collected_data')

    X_all, Y_all, S_all, split_all = [], [], [], []
    session_dirs = [os.path.join(data_dir, 'train'), os.path.join(data_dir, 'test'), data_dir]
    bases = []
    for root in session_dirs:
        for b in sorted(glob.glob(os.path.join(root, 'gesture_data_*.raw'))):
            split = 'test' if root.endswith('test') else 'train'
            bases.append((b[:-len('.raw')], split))
    for si, (base, split) in enumerate(bases):
        samples, meta, events = load_session(base)
        L = int(meta['chirp_length'])
        P = load_or_compute_profile(base, samples, meta)
        n_frames = P.shape[1]
        y_tl = frame_labels(events, n_frames, L)
        print(f'{os.path.basename(base)}: {n_frames} frames')

        wins = []
        for f0 in range(0, n_frames - wf + 1, sf):
            wins.append((make_window(P, f0, wf),
                         y_tl[f0:f0 + wf].copy()))

        for w, yw in wins:
            X_all.append(w)
            Y_all.append(yw)
            S_all.append(si)
            split_all.append(split)
        print(f'  [{split}] windows: {len(wins)}')

    np.savez(out_path,
             X=np.stack(X_all),
             Y=np.stack(Y_all).astype(np.int64),
             session=np.array(S_all),
             split=np.array(split_all))
    print(f'saved {out_path}: X {np.stack(X_all).shape}, '
          f'label mix {np.bincount(np.concatenate(Y_all)).tolist()}')


if __name__ == '__main__':
    main()
