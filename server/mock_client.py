import socket
import struct
import time
import numpy as np

"""
Mock client that connects to WatchHand Java Server and sends random PCM data.
Tests the server's data processing pipeline.
"""

SERVER_HOST = '127.0.0.1'
SERVER_PORT = 9999
SAMPLE_RATE = 44100
CHIRP_LENGTH = round(SAMPLE_RATE * 4.0 / 300.0)  # 统一 chirp 时长 13.333ms（4/300s）
F_MIN = 18000.0
F_MAX = 20000.0

def generate_chirp():
    """Generate one FMCW chirp with Hann window."""
    t = np.arange(CHIRP_LENGTH) / SAMPLE_RATE
    T = CHIRP_LENGTH / SAMPLE_RATE
    phase = 2 * np.pi * (F_MIN * t + (F_MAX - F_MIN) * t**2 / (2 * T))
    chirp = np.cos(phase)
    hann = 0.5 * (1 - np.cos(2 * np.pi * np.arange(CHIRP_LENGTH) / (CHIRP_LENGTH - 1)))
    return (chirp * hann * 10000).astype(np.int16)

def main():
    print(f"Connecting to {SERVER_HOST}:{SERVER_PORT}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((SERVER_HOST, SERVER_PORT))
    print("Connected!")

    # Send header
    sock.sendall(b"WATCHHAND\n")
    sock.sendall(struct.pack('<i', SAMPLE_RATE))
    sock.sendall(bytes([1, 16]))  # mono, 16-bit
    print("Header sent: WATCHHAND, 44100Hz, mono, 16-bit")

    # Generate chirp
    chirp = generate_chirp()
    print(f"Chirp generated: {len(chirp)} samples")

    # Build a buffer of repeated chirps (3 seconds)
    duration = 3.0
    total_samples = int(SAMPLE_RATE * duration)
    n_chirps = total_samples // CHIRP_LENGTH
    
    buffer = np.zeros(total_samples, dtype=np.int16)
    for i in range(n_chirps):
        start = i * CHIRP_LENGTH
        # Direct path (strong)
        delay = 20
        if start + delay + CHIRP_LENGTH < total_samples:
            buffer[start + delay:start + delay + CHIRP_LENGTH] += chirp
        # Echo (weaker, varying)
        echo_delay = 100 + int(30 * np.sin(2 * np.pi * i / max(n_chirps, 1)))
        if start + echo_delay + CHIRP_LENGTH < total_samples:
            buffer[start + echo_delay:start + echo_delay + CHIRP_LENGTH] += (chirp * 0.3).astype(np.int16)
    
    # Add noise
    buffer = buffer + np.random.randint(-200, 200, total_samples, dtype=np.int16)
    
    print(f"Sending {total_samples} samples ({duration}s) in 50ms chunks...")
    
    chunk_samples = 2205  # 50ms @ 44.1kHz
    for i in range(0, total_samples, chunk_samples):
        chunk = buffer[i:i+chunk_samples]
        sock.sendall(struct.pack(f'<{len(chunk)}h', *chunk))
        time.sleep(0.05)
    
    print(f"Sent all data. Waiting 5 seconds for processing...")
    time.sleep(5)
    
    sock.close()
    print("Disconnected.")

if __name__ == '__main__':
    main()
