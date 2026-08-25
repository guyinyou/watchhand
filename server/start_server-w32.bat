@echo off
REM WatchHand Java Server startup script (Windows)
REM Usage: start_server.bat [port] [model.onnx]

cd /d "%~dp0"

set JAVA_HOME=D:\JDK\openjdk-21.0.2
set PATH=%JAVA_HOME%\bin;%PATH%

set PORT=%1
set MODEL=%2
if "%PORT%"=="" set PORT=9999
if "%MODEL%"=="" set MODEL=train\last_w32.onnx

echo Compiling WatchHandServer.java ...
javac -cp "lib\*" WatchHandServer.java
if errorlevel 1 (
    echo COMPILE FAILED
    pause
    exit /b 1
)
echo COMPILE_OK

echo Starting WatchHand Java Server on port %PORT%...
echo Model: %MODEL%

java -cp ".;lib\*" WatchHandServer %PORT% %MODEL%
pause
