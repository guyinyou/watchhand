#!/bin/bash
# WatchHand Android APK 打包脚本
#
# 用法:
#   ./build_apk.sh                 # 仅构建 debug APK
#   ./build_apk.sh --install       # 构建并安装到已连接设备
#   ./build_apk.sh --install <serial>  # 构建并安装到指定设备
#   ./build_apk.sh --release       # 构建 release 包（未配置签名时输出未签名包）

set -e
cd "$(dirname "$0")"

# --- JDK 17 环境（AGP 8.x 要求）---
if [ -z "$JAVA_HOME" ] || ! "$JAVA_HOME/bin/java" -version 2>&1 | grep -q 'version "17'; then
    for candidate in \
        /Library/Java/JavaVirtualMachines/jdk-17.jdk/Contents/Home \
        /Library/Java/JavaVirtualMachines/amazon-corretto-17.jdk/Contents/Home \
        /usr/local/opt/openjdk@17; do
        if [ -x "$candidate/bin/java" ]; then
            export JAVA_HOME="$candidate"
            break
        fi
    done
fi
if [ -z "$JAVA_HOME" ]; then
    echo "错误: 未找到 JDK 17，请手动 export JAVA_HOME 后重试" >&2
    exit 1
fi
echo "JAVA_HOME=$JAVA_HOME"

# --- 参数解析 ---
INSTALL=false
DEVICE=""
TASK=":app:assembleDebug"
for arg in "$@"; do
    case "$arg" in
        --install) INSTALL=true ;;
        --release) TASK=":app:assembleRelease" ;;
        *) DEVICE="$arg" ;;
    esac
done

# --- 构建 ---
echo "开始构建: gradle $TASK"
gradle "$TASK" --no-daemon

APK_DIR="app/build/outputs/apk"
APK=$(ls -t "$APK_DIR"/debug/*.apk "$APK_DIR"/release/*.apk 2>/dev/null | head -1)
if [ -z "$APK" ]; then
    echo "错误: 未找到生成的 APK" >&2
    exit 1
fi
echo ""
echo "构建成功: $APK ($(du -h "$APK" | cut -f1))"

# --- 安装 ---
if $INSTALL; then
    if [ -n "$DEVICE" ]; then
        echo "安装到设备 $DEVICE ..."
        adb -s "$DEVICE" install -r "$APK"
    else
        echo "安装到已连接设备 ..."
        adb install -r "$APK"
    fi
    echo "安装完成"
fi
