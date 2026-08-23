#!/usr/bin/env python3
"""对比流式分段相关（Java processChunk 同构模拟）与 extract 批处理轮廓的对齐约定。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from watchhand_server import EchoProfileProcessor, BiquadFilter
import extract as E

base = sys.argv[1] if len(sys.argv) > 1 else \
    '../collected_data/test/gesture_data_20260819_124247'
samples, meta, events = E.load_session(base)
fs = int(meta['sample_rate']); L = int(meta['chirp_length'])
B = int(meta['distance_bins']); T = int(meta['time_window_frames'])

proc = E.EchoProfileProcessor if False else EchoProfileProcessor(
    fs, float(meta['f_min']), float(meta['f_max']), L, B, T)
x = samples.astype(np.float32)
for _ in range(3):
    x = proc._biquad_filter_vectorized(x, proc.hp_coeffs, BiquadFilter())
for _ in range(2):
    x = proc._biquad_filter_vectorized(x, proc.lp_coeffs, BiquadFilter())

corr0 = np.correlate(x[:4 * L], proc.tx_chirp, 'full')
bestIdx = int(np.argmax(np.abs(corr0[:3 * L])))

P = E.batch_profile(samples, meta)
Praw = E.batch_profile(samples, meta)


def stream_sim(off):
    cols = []
    n = (len(x) // L) * L
    for f in range(2, n // L):
        seg = x[(f - 2) * L:(f + 1) * L]
        corr = np.correlate(seg, proc.tx_chirp, 'full')
        cols.append(corr[(L - 1) + off:(L - 1) + off + B])
    return np.array(cols).T[:, -T:]


cands = [('paper(L/2)', (bestIdx + L - L // 2) % L),
         ('b0-5', (bestIdx - 5) % L),
         ('b0-6', (bestIdx - 6) % L)]
# extract 的 p_start 逻辑（在 alignBuf 的 corr 上复现）
corrA = np.correlate(x[:4 * L], proc.tx_chirp, 'full')
n0 = (len(corrA) // L) * L
b0 = int(np.argmax(np.abs(corrA[:n0]).reshape(-1, L).mean(axis=0)))
p_start = None
for cand in [(b0 - 5) % L, (5 - b0) % L]:
    nf = (len(corrA) - cand) // L - 2
    if nf <= 0:
        continue
    fr = corrA[cand:cand + nf * L].reshape(nf, L)
    if 0 <= int(np.argmax(np.abs(fr[:, :B]).mean(axis=0))) - 5 <= 2:
        p_start = cand
        break
if p_start is None:
    p_start = (b0 - 5) % L
cands.append(('p_start+1', (p_start + 1) % L))
for name, off in cands:
    S = stream_sim(off)
    rel = np.abs(S - Praw[:, -T:]).max() / np.abs(Praw[:, -T:]).max()
    print(f'{name}: rel diff={rel:.3f} peak_bin={int(np.argmax(np.abs(S).mean(1)))} '
          f'amp={np.abs(S).mean():.1f}')
print('batch(raw): peak_bin=5 amp=', round(float(np.abs(Praw[:, -T:]).mean()), 1))
