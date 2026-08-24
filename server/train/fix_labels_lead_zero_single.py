"""一次性修复：单手势会话开头的 0 标签段整体改为该类标签。

判定规则（用户确认）：去掉开头连续的 label=0 事件后，剩余事件
全部为同一类 N（N!=0，且直到文件末尾无其他类、无回 0），
则视为单手势 N 会话，开头 0 段是引导录制准备期遗留，
压缩为单个 (0, N) 事件。打印每个被修文件供人工核对。
"""
import glob
import os

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'collected_data')
changed = 0
for f in sorted(glob.glob(os.path.join(root, '*', '*.labels'))):
    lines = [l.strip() for l in open(f) if l.strip()]
    evs = [tuple(map(int, l.split())) for l in lines]
    # 找第一个非 0 事件
    k = next((i for i, (_, lab) in enumerate(evs) if lab != 0), None)
    if k is None or k == 0:
        continue                                # 全 0 或开头没有 0 段
    rest = evs[k:]
    labels = {lab for _, lab in rest}
    if len(labels) != 1:
        continue                                # 后续不止一类，不是单手势会话
    n = rest[0][1]
    new_evs = [(0, n)] + rest[1:]
    with open(f, 'w') as out:
        out.write('\n'.join(f'{o} {l}' for o, l in new_evs) + '\n')
    changed += 1
    print(f'fixed {os.path.basename(f)}: 开头 {k} 个 0 事件 -> (0, {n}), '
          f'共 {len(new_evs)} 个事件')
print(f'done: {changed} files fixed')
