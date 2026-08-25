#!/usr/bin/env python3
"""
把 last.pt 导出为 ONNX（供 Java 服务端 ONNX Runtime 实时推理）。

输入 : x  float32 [1, 2, 64, 96]（双通道回声轮廓，距离轴 pad 到 64）
输出 : logits  float32 [1, 12, K]（密集时间头，最后取末步 argmax 显示）

Usage (project python 3.10):
  python3 export_onnx.py [--last last.pt] [--out last.onnx]
"""

import argparse

import torch

from train import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--last', default='last.pt')
    ap.add_argument('--out', default='last.onnx')
    args = ap.parse_args()

    ck = torch.load(args.last, map_location='cpu', weights_only=True)
    # last.pt 已含完整权重，pretrained=False 跳过预训练下载
    model = build_model(ck.get('arch', 'transformer'), int(ck['num_classes']),
                        d_model=ck.get('dmodel', 256),
                        layers=ck.get('layers', 4),
                        dropout=ck.get('dropout', 0.2),
                        pretrained=False)
    model.load_state_dict(ck['model'])
    model.eval()

    x = torch.randn(1, 2, 64, 96)
    torch.onnx.export(model, x, args.out,
                      input_names=['x'], output_names=['logits'],
                      opset_version=14, dynamic_axes={'x': {0: 'batch'},
                                                      'logits': {0: 'batch'}})
    print(f'exported {args.out} (epoch {ck.get("epoch", "?")}, '
          f'K={int(ck["num_classes"])}, dmodel={ck.get("dmodel", 256)}, '
          f'layers={ck.get("layers", 4)})')


if __name__ == '__main__':
    main()
