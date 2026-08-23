#!/bin/bash
# WatchHand Java Server startup script
# Usage: ./start_server.sh [port] [model.onnx]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JAVA_HOME="/Library/Java/JavaVirtualMachines/jdk-17.jdk/Contents/Home"
PORT="${1:-9999}"
MODEL="${2:-train/last.onnx}"


cd /Users/guyinyou/qoderAgent/watchhand-android/server && /Library/Java/JavaVirtualMachines/jdk-17.jdk/Contents/Home/bin/javac -cp "lib/*" WatchHandServer.java && echo COMPILE_OK



echo "Starting WatchHand Java Server on port $PORT..."
echo "Java: $JAVA_HOME/bin/java"
echo "Model: $MODEL"

cd "$SCRIPT_DIR"
exec "$JAVA_HOME/bin/java" -cp ".:lib/*" WatchHandServer "$PORT" "$MODEL"
