#!/usr/bin/env python3
"""
Test TCP server with mock client. Sends mock PCM data via TCP.
"""

import socket
import struct
import threading
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from watchhand_server import WatchHandServer

SAMPLE_RATE = 44100
SERVER_HOST = '127.0.0.1'
SERVER_PORT = 19999  # Use different port to avoid conflicts

def mock_client():
    """Connect as a mock Android client and send PCM data."""
    time.sleep(1)  # Wait for server to start
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((SERVER_HOST, SERVER_PORT))
    
    # Send header
    sock.sendall(b"WATCHHAND\n")
    sock.sendall(struct.pack('<i', SAMPLE_RATE))
    sock.sendall(bytes([1, 16]))  # mono, 16-bit
    
    print("[Mock Client] Connected and header sent")
    
    # Generate and send mock PCM data
    import numpy as np
    from watchhand_server import F_MIN, F_MAX, CHIRP_LENGTH
    
    duration = 3.0
    total_samples = int(SAMPLE_RATE * duration)
    samples = np.zeros(total_samples, dtype=np.int16)
    
    n_chirps = total_samples // CHIRP_LENGTH
    for i in range(n_chirps):
        chirp_start = i * CHIRP_LENGTH
        chirp_t = np.arange(CHIRP_LENGTH) / SAMPLE_RATE
        T = CHIRP_LENGTH / SAMPLE_RATE
        phase = 2 * np.pi * (F_MIN * chirp_t + (F_MAX - F_MIN) * chirp_t**2 / (2 * T))
        chirp = np.cos(phase)
        hann = 0.5 * (1 - np.cos(2 * np.pi * np.arange(CHIRP_LENGTH) / (CHIRP_LENGTH - 1)))
        chirp_windowed = (chirp * hann * 10000).astype(np.int16)
        
        # Direct path
        delay = 20
        if chirp_start + delay + CHIRP_LENGTH < total_samples:
            samples[chirp_start + delay:chirp_start + delay + CHIRP_LENGTH] += chirp_windowed
        
        # Echo
        echo_delay = 100 + int(30 * np.sin(2 * np.pi * i / max(n_chirps, 1)))
        if chirp_start + echo_delay + CHIRP_LENGTH < total_samples:
            samples[chirp_start + echo_delay:chirp_start + echo_delay + CHIRP_LENGTH] += (chirp_windowed * 0.3).astype(np.int16)
    
    # Send in 100ms chunks
    chunk_size = 4800
    for i in range(0, len(samples), chunk_size):
        chunk = samples[i:i+chunk_size]
        sock.sendall(struct.pack(f'<{len(chunk)}h', *chunk))
        time.sleep(0.1)
    
    print(f"[Mock Client] Sent {len(samples)} samples")
    sock.close()
    print("[Mock Client] Disconnected")


def main():
    print("=" * 60)
    print("WatchHand TCP Server Test")
    print("=" * 60)
    
    # Start server in background thread
    server = WatchHandServer(host=SERVER_HOST, port=SERVER_PORT)
    server.running = True
    
    # Start mock client
    client_thread = threading.Thread(target=mock_client, daemon=True)
    client_thread.start()
    
    # Accept connection and process
    server.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.server_socket.bind((SERVER_HOST, SERVER_PORT))
    server.server_socket.listen(1)
    server.server_socket.settimeout(5)
    
    print(f"\n[Server] Listening on {SERVER_HOST}:{SERVER_PORT}")
    
    try:
        server.client_socket, addr = server.server_socket.accept()
        print(f"[Server] Connected: {addr}")
        server.start_time = time.time()
        
        server._receive_header()
        print("[Server] Header received")
        
        # Process data
        server._process_loop()
        
        elapsed = time.time() - server.start_time
        print(f"\n[Server] Processed {server.total_samples_received} samples in {elapsed:.1f}s")
        print(f"[Server] Frames produced: {server.total_frames_processed}")
        
        if server.total_frames_processed > 0:
            print("\nTEST PASSED - TCP pipeline working correctly")
        else:
            print("\nTEST FAILED - No frames produced")
    except socket.timeout:
        print("\nTEST FAILED - Connection timeout")
    except Exception as e:
        print(f"\nTEST FAILED - {e}")
    finally:
        server.stop()


if __name__ == '__main__':
    main()
