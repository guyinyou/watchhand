"""一次性修复：删除 .labels 开头 '0 N' 之后误插入的 '0 0' 行。

背景：旧版引导录制开始时残留 currentLabel 写出 {0,N}，随后又强制记录
准备期 {0,0}，导致开头 N->0->N 的错误标签序列。本脚本删除该 0 行，
让开头的 N 标签保持连续。幂等，可重复执行。
"""
import glob
import os

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'collected_data')
changed = 0
for f in sorted(glob.glob(os.path.join(root, '*', '*.labels'))):
    lines = [l.strip() for l in open(f) if l.strip()]
    evs = [tuple(map(int, l.split())) for l in lines]
    if len(evs) >= 2 and evs[0][0] == 0 and evs[0][1] != 0 and evs[1] == (0, 0):
        with open(f, 'w') as out:
            out.write('\n'.join([lines[0]] + lines[2:]) + '\n')
        changed += 1
        print('fixed', os.path.basename(f))
print(f'done: {changed} files fixed')
