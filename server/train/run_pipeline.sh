#!/bin/bash
# 一键迭代：提取 -> 训练(自动续训) -> 测试 -> 导出 ONNX
#
# Usage:
#   ./run_pipeline.sh                 # 默认 80 epoch
#   EPOCHS=200 ./run_pipeline.sh      # 指定 epoch 数
#   ./run_pipeline.sh --reset         # 额外参数透传给 train.py（如 --reset 从头训）

set -e
cd "$(dirname "$0")"

PY=/Library/Frameworks/Python.framework/Versions/3.10/bin/python3
EPOCHS=${EPOCHS:-80}

echo "================ extract ================"
$PY extract.py

echo "================ train ================="
$PY -u train.py --epochs "$EPOCHS" --device mps \
    --layers 2 --dmodel 128 --batch 32 --exclude-session 0 "$@"

echo "================ test =================="
$PY test.py --device mps

echo "================ export onnx ============"
$PY export_onnx.py

echo "完成。重启 server (./start_server.sh) 使新模型上线"
