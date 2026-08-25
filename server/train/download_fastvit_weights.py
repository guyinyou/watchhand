"""经代理下载 fastvit_t12 ImageNet 预训练权重到本地（只需运行一次）。

用法（先在同一窗口设置代理，端口按实际改）：
  $env:HTTPS_PROXY='http://127.0.0.1:7897'
  $env:HTTP_PROXY='http://127.0.0.1:7897'
  python download_fastvit_weights.py

下载成功后 train.py 会自动加载本地权重，不再需要网络。
"""
import os
import urllib.request

URL = 'https://huggingface.co/timm/fastvit_t12.apple_in1k/resolve/main/model.safetensors'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'fastvit_t12_in1k.safetensors')

if os.path.exists(OUT):
    print(f'已存在，无需下载: {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB)')
    raise SystemExit(0)

print('下载:', URL)
print('代理:', os.environ.get('HTTPS_PROXY', '未设置（直连）'))
tmp = OUT + '.tmp'
urllib.request.urlretrieve(URL, tmp)          # urllib 自动读 HTTPS_PROXY 环境变量
size = os.path.getsize(tmp) / 1e6
if size < 20:
    os.remove(tmp)
    raise SystemExit(f'下载失败：文件仅 {size:.1f} MB（正常约 25MB），请检查代理')
os.replace(tmp, OUT)
print(f'完成: {OUT} ({size:.1f} MB)')
