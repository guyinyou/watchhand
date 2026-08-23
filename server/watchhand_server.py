#!/usr/bin/env python3
"""
WatchHand TCP Server - Receives raw PCM audio from Android device
and performs real-time echo profile processing and visualization.

Usage:
    python3 watchhand_server.py [--host 0.0.0.0] [--port 9999]
"""

import argparse
import socket
import struct
import threading
import numpy as np
from numpy.fft import fft, ifft

# 直达声锚定 bin，与 train/extract.py 的 DIRECT_BIN 保持一致
DIRECT_BIN = 5
import matplotlib
# Try to use a GUI backend, fall back to Agg if not available
try:
    import tkinter
    matplotlib.use('TkAgg')
except ImportError:
    try:
        import PyQt5
        matplotlib.use('Qt5Agg')
    except ImportError:
        try:
            import PyQt6
            matplotlib.use('Qt6Agg')
        except ImportError:
            matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time
import sys

# Signal processing parameters
F_MIN = 18000.0
F_MAX = 20000.0
CHIRP_LENGTH = round(44100 * 4.0 / 300.0)  # 统一 chirp 时长 13.333ms（4/300s），按采样率换算
DISTANCE_BINS = 60
TIME_WINDOW_FRAMES = 96


class BiquadFilter:
    """Biquad filter state for streaming processing."""
    def __init__(self):
        self.z1 = 0.0
        self.z2 = 0.0


class EchoProfileProcessor:
    """Python implementation of the echo profile processing pipeline."""
    
    def __init__(self, sample_rate, f_min, f_max, chirp_length, distance_bins, time_window_frames):
        self.sample_rate = sample_rate
        self.f_min = f_min
        self.f_max = f_max
        self.chirp_length = chirp_length
        self.distance_bins = distance_bins
        self.time_window_frames = time_window_frames
        
        # Generate TX chirp with Hann window
        self.tx_chirp = self._generate_chirp()
        
        # Biquad filter coefficients
        self.hp_coeffs = self._compute_highpass_coeffs(f_min, sample_rate)
        self.lp_coeffs = self._compute_lowpass_coeffs(f_max, sample_rate)
        
        # Filter states (persist across feed calls)
        self.hp_states = [BiquadFilter() for _ in range(3)]
        self.lp_states = [BiquadFilter() for _ in range(2)]
        
        # Ring buffer for filtered audio
        self.process_chirps = time_window_frames + 30
        self.rx_buffer_size = self.process_chirps * chirp_length
        self.rx_ring = np.zeros(self.rx_buffer_size, dtype=np.float32)
        self.rx_head = 0
        self.rx_filled = 0
        
        # Start alignment
        self.start_offset = -1
        self.align_buf = np.zeros(chirp_length * 4, dtype=np.float32)
        self.align_pos = 0
        self.start_found = False
        self.align_pass = 0  # SNR 连续通过计数：满 8 个 chirp 边界才锁定
    
    def _generate_chirp(self):
        """Generate FMCW chirp signal with Hann window."""
        t = np.arange(self.chirp_length) / self.sample_rate
        T = self.chirp_length / self.sample_rate
        phase = 2 * np.pi * (self.f_min * t + (self.f_max - self.f_min) * t**2 / (2 * T))
        chirp = np.cos(phase).astype(np.float32)
        # Apply Hann window to reduce spectral leakage
        hann_window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(self.chirp_length) / (self.chirp_length - 1)))
        return chirp * hann_window
    
    def _compute_highpass_coeffs(self, fc, fs):
        """Compute highpass biquad coefficients."""
        w0 = 2 * np.pi * fc / fs
        cos_w0 = np.cos(w0)
        sin_w0 = np.sin(w0)
        alpha = sin_w0 / (2 * 0.707)
        a0 = 1 + alpha
        
        b0 = (1 + cos_w0) / 2 / a0
        b1 = -(1 + cos_w0) / a0
        b2 = (1 + cos_w0) / 2 / a0
        a1 = -2 * cos_w0 / a0
        a2 = (1 - alpha) / a0
        
        return np.array([b0, b1, b2, a1, a2], dtype=np.float32)
    
    def _compute_lowpass_coeffs(self, fc, fs):
        """Compute lowpass biquad coefficients."""
        w0 = 2 * np.pi * fc / fs
        cos_w0 = np.cos(w0)
        sin_w0 = np.sin(w0)
        alpha = sin_w0 / (2 * 0.707)
        a0 = 1 + alpha
        
        b0 = (1 - cos_w0) / 2 / a0
        b1 = (1 - cos_w0) / a0
        b2 = (1 - cos_w0) / 2 / a0
        a1 = -2 * cos_w0 / a0
        a2 = (1 - alpha) / a0
        
        return np.array([b0, b1, b2, a1, a2], dtype=np.float32)
    
    def _biquad_filter_vectorized(self, x, coeffs, state):
        """Apply biquad filter to entire array at once."""
        b0, b1, b2, a1, a2 = coeffs
        n = len(x)
        y = np.zeros(n, dtype=np.float32)
        z1 = state.z1
        z2 = state.z2
        
        for i in range(n):
            y[i] = b0 * x[i] + z1
            z1 = b1 * x[i] - a1 * y[i] + z2
            z2 = b2 * x[i] - a2 * y[i]
        
        state.z1 = z1
        state.z2 = z2
        return y
    
    def _find_start_offset(self, buf):
        """Find start position using cross-correlation (vectorized).
        与 extract.py batch_profile 的 p_start 约定一致：直达声锚定 DIRECT_BIN，
        直接返回 p_start（流式 feed 从该 lag 起切帧）。"""
        corr = self._cross_correlate_full(buf, self.tx_chirp)
        search_len = min(len(corr), self.chirp_length * 3)
        # SNR 门槛：直达声峰值需显著高于相关底噪（实测信号 ≈14，静默 ≈5）
        m = np.mean(np.abs(corr[:search_len]))
        if m <= 0 or np.max(np.abs(corr[:search_len])) < 9 * m:
            return None
        L = self.chirp_length
        n0 = (search_len // L) * L
        b0 = int(np.argmax(np.abs(corr[:n0]).reshape(-1, L).mean(axis=0)))
        p_start = (b0 - DIRECT_BIN + L) % L
        for cand in [(b0 - DIRECT_BIN) % L, (DIRECT_BIN - b0) % L]:
            nf = (search_len - cand) // L - 2
            if nf <= 0:
                continue
            fr = corr[cand:cand + nf * L].reshape(nf, L)
            if 0 <= int(np.argmax(np.abs(fr[:, :self.distance_bins]).mean(axis=0))) - DIRECT_BIN <= 2:
                p_start = cand
                break
        return p_start % L
    
    def _cross_correlate_full(self, rx, tx):
        """Full cross-correlation using FFT (much faster)."""
        n = len(rx) + len(tx) - 1
        needed = self.process_chirps * self.chirp_length
        compute_len = min(n, needed)
        
        # FFT-based correlation
        fft_size = 1
        while fft_size < n:
            fft_size *= 2
        
        rx_fft = fft(rx, fft_size)
        tx_fft = fft(tx, fft_size)
        corr = np.real(ifft(rx_fft * np.conj(tx_fft)))[:compute_len]
        
        return corr.astype(np.float32)
    
    def feed(self, samples):
        """Feed raw audio samples. Returns (original_profile, diff_profile, frame_count) or None."""
        # Convert to float array
        x = np.array(samples, dtype=np.float32)
        
        # Apply bandpass filter (vectorized)
        for state in self.hp_states:
            x = self._biquad_filter_vectorized(x, self.hp_coeffs, state)
        for state in self.lp_states:
            x = self._biquad_filter_vectorized(x, self.lp_coeffs, state)
        
        if not self.start_found:
            # 缓冲满后按 chirp 边界重试对齐：信号未到（SNR 不足）就继续等，
            # 避免“先连接后开音”时锁定在纯噪声上
            i = 0
            while i < len(x):
                self.align_buf[self.align_pos % len(self.align_buf)] = x[i]
                self.align_pos += 1
                i += 1
                if self.align_pos >= len(self.align_buf) and self.align_pos % self.chirp_length == 0:
                    off = self._find_start_offset(self.align_buf)
                    if off is not None:
                        # 信号刚到时缓冲只有一部分是信号，需连续 8 次通过才锁定
                        self.align_pass += 1
                        if self.align_pass >= 8:
                            self.start_offset = off
                            self.start_found = True
                            break
                    else:
                        self.align_pass = 0
            if not self.start_found:
                return None
            self.rx_head = 0
            self.rx_filled = 0
            for s in x[i:]:
                self.rx_ring[self.rx_head % len(self.rx_ring)] = s
                self.rx_head += 1
                if self.rx_filled < len(self.rx_ring):
                    self.rx_filled += 1
            return None
        
        # Add to ring buffer
        for s in x:
            self.rx_ring[self.rx_head % len(self.rx_ring)] = s
            self.rx_head += 1
            if self.rx_filled < len(self.rx_ring):
                self.rx_filled += 1
        
        # Process when enough data
        if self.rx_filled < self.rx_buffer_size or not self.start_found:
            return None
        
        # Extract contiguous window (vectorized)
        extract_size = self.rx_buffer_size
        start_idx = ((self.rx_head - extract_size) % len(self.rx_ring) + len(self.rx_ring)) % len(self.rx_ring)
        
        if start_idx + extract_size <= len(self.rx_ring):
            rx_data = self.rx_ring[start_idx:start_idx + extract_size].copy()
        else:
            # Wrap around
            first_part = self.rx_ring[start_idx:]
            second_part = self.rx_ring[:extract_size - len(first_part)]
            rx_data = np.concatenate([first_part, second_part])
        
        # Apply start alignment
        aligned_rx = rx_data[max(0, min(self.start_offset, len(rx_data) - 1)):]
        
        # Full cross-correlation (FFT-based)
        corr = self._cross_correlate_full(aligned_rx, self.tx_chirp)
        
        # Reshape into frames
        n_frames = len(corr) // self.chirp_length
        if n_frames < self.time_window_frames:
            return None
        
        # Build original profile (vectorized, column-major)
        frame_start = n_frames - self.time_window_frames
        frames_data = corr[frame_start * self.chirp_length:(frame_start + self.time_window_frames) * self.chirp_length]
        frames_2d = frames_data.reshape(self.time_window_frames, self.chirp_length)
        original_profile = frames_2d[:, :self.distance_bins].T.flatten()
        
        # Differential profile (vectorized)
        abs_profile = np.abs(frames_2d[:, :self.distance_bins].T)  # shape: (distance_bins, time_window_frames)
        diff_profile = (abs_profile[:, 1:] - abs_profile[:, :-1]).flatten()
        
        return original_profile, diff_profile, self.time_window_frames


class WatchHandServer:
    """TCP server that receives PCM audio and visualizes echo profiles in real-time."""
    
    def __init__(self, host='0.0.0.0', port=9999):
        self.host = host
        self.port = port
        self.server_socket = None
        self.client_socket = None
        self.processor = None
        self.running = False
        
        # Visualization data (shared between threads)
        self.original_profile = None
        self.diff_profile = None
        self.frame_count = 0
        self.data_lock = threading.Lock()
        
        # Statistics
        self.total_samples_received = 0
        self.total_frames_processed = 0
        self.start_time = None
        
        # Plot
        self.fig = None
        self.ax_orig = None
        self.ax_diff = None
    
    def start(self):
        """Start the TCP server."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        self.running = True
        
        print(f"WatchHand Server listening on {self.host}:{self.port}")
        print("Waiting for Android device connection...")
        
        # Accept connection
        self.client_socket, addr = self.server_socket.accept()
        print(f"Connected: {addr}")
        self.start_time = time.time()
        
        # Receive header
        self._receive_header()
        
        # Setup visualization
        self._setup_visualization()
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()
        
        # Start visualization update loop
        self._run_visualization()
    
    def _receive_header(self):
        """Receive and parse connection header."""
        # Read "WATCHHAND\n"
        header = b""
        while not header.endswith(b"\n"):
            byte = self.client_socket.recv(1)
            if not byte:
                break
            header += byte
        
        print(f"Header: {header.strip().decode()}")
        
        if header.strip() != b"WATCHHAND":
            raise ValueError(f"Invalid header: {header}")
        
        # Read sample rate (4 bytes, little-endian)
        sample_rate_bytes = self._recv_exact(4)
        sample_rate = struct.unpack('<i', sample_rate_bytes)[0]
        
        # Read channels (1 byte) and bits per sample (1 byte)
        config_bytes = self._recv_exact(2)
        channels = config_bytes[0]
        bits_per_sample = config_bytes[1]
        
        print(f"Sample rate: {sample_rate} Hz")
        print(f"Channels: {channels}, Bits per sample: {bits_per_sample}")
        
        # Initialize processor
        self.processor = EchoProfileProcessor(
            sample_rate=sample_rate,
            f_min=F_MIN,
            f_max=F_MAX,
            chirp_length=CHIRP_LENGTH,
            distance_bins=DISTANCE_BINS,
            time_window_frames=TIME_WINDOW_FRAMES
        )
    
    def _recv_exact(self, n):
        """Receive exactly n bytes."""
        data = b""
        while len(data) < n:
            chunk = self.client_socket.recv(n - len(data))
            if not chunk:
                break
            data += chunk
        return data
    
    def _process_loop(self):
        """Main processing loop - runs in background thread."""
        buffer = b""
        samples_per_read = 4410  # 100ms at 44.1kHz
        
        print("Processing thread started")
        
        while self.running:
            try:
                data = self.client_socket.recv(samples_per_read * 2 * 4)  # Read more at once
                if not data:
                    print("Client disconnected")
                    break
                
                self.total_samples_received += len(data) // 2
                buffer += data
                
                # Process when we have enough samples
                while len(buffer) >= samples_per_read * 2:
                    # Extract samples
                    sample_bytes = buffer[:samples_per_read * 2]
                    buffer = buffer[samples_per_read * 2:]
                    
                    # Convert to short array
                    samples = struct.unpack(f'<{samples_per_read}h', sample_bytes)
                    samples_array = np.array(samples, dtype=np.float32)
                    
                    # Feed to processor
                    result = self.processor.feed(samples_array)
                    if result is not None:
                        orig, diff, frames = result
                        with self.data_lock:
                            self.original_profile = orig
                            self.diff_profile = diff
                            self.frame_count = frames
                        self.total_frames_processed += 1
                        
                        # Print statistics every 10 frames
                        if self.total_frames_processed % 10 == 0:
                            elapsed = time.time() - self.start_time
                            print(f"[{elapsed:.1f}s] Frames: {self.total_frames_processed}, "
                                  f"Samples: {self.total_samples_received}, "
                                  f"Rate: {self.total_samples_received / elapsed / 1000:.1f}k samples/s")
                
            except Exception as e:
                print(f"Processing error: {e}")
                break
        
        print("Processing thread stopped")
        self.running = False
    
    def _setup_visualization(self):
        """Setup matplotlib visualization."""
        self.fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(2, 1, height_ratios=[1, 1])
        
        self.ax_orig = self.fig.add_subplot(gs[0])
        self.ax_orig.set_title('Original Echo Profile')
        self.ax_orig.set_xlabel('Frame')
        self.ax_orig.set_ylabel('Distance Bin')
        
        self.ax_diff = self.fig.add_subplot(gs[1])
        self.ax_diff.set_title('Differential Echo Profile')
        self.ax_diff.set_xlabel('Frame')
        self.ax_diff.set_ylabel('Distance Bin')
        
        plt.tight_layout()
        
        # Check if we have an interactive backend
        self.interactive = matplotlib.is_interactive() or plt.get_backend().lower() not in ['agg', 'pdf', 'svg', 'ps']
    
    def _run_visualization(self):
        """Real-time visualization update loop."""
        print("Visualization started")
        print(f"Backend: {plt.get_backend()}, Interactive: {self.interactive}")
        
        frame_counter = 0
        
        while self.running:
            # Check for new data
            with self.data_lock:
                if self.original_profile is not None and self.diff_profile is not None:
                    orig_2d = self.original_profile.reshape(DISTANCE_BINS, TIME_WINDOW_FRAMES)
                    diff_2d = self.diff_profile.reshape(DISTANCE_BINS, TIME_WINDOW_FRAMES - 1)
            
            # Update plots
            self.ax_orig.clear()
            self.ax_diff.clear()
            
            if self.original_profile is not None:
                # Percentile clipping
                orig_lo = np.percentile(orig_2d, 2)
                orig_hi = np.percentile(orig_2d, 98)
                orig_clipped = np.clip(orig_2d, orig_lo, orig_hi)
                
                self.ax_orig.imshow(orig_clipped, aspect='auto', cmap='coolwarm',
                                   extent=[0, TIME_WINDOW_FRAMES, DISTANCE_BINS, 0])
                self.ax_orig.set_title(f'Original Echo Profile (Range: {orig_lo:.2e} ~ {orig_hi:.2e})')
            else:
                self.ax_orig.text(0.5, 0.5, 'Waiting for data...', 
                                 ha='center', va='center', transform=self.ax_orig.transAxes)
                self.ax_orig.set_title('Original Echo Profile')
            
            self.ax_orig.set_xlabel('Frame')
            self.ax_orig.set_ylabel('Distance Bin')
            
            if self.diff_profile is not None:
                diff_lo = np.percentile(diff_2d, 2)
                diff_hi = np.percentile(diff_2d, 98)
                diff_clipped = np.clip(diff_2d, diff_lo, diff_hi)
                
                self.ax_diff.imshow(diff_clipped, aspect='auto', cmap='coolwarm',
                                   extent=[0, TIME_WINDOW_FRAMES - 1, DISTANCE_BINS, 0])
                self.ax_diff.set_title(f'Differential Echo Profile (Range: {diff_lo:.2e} ~ {diff_hi:.2e})')
            else:
                self.ax_diff.text(0.5, 0.5, 'Waiting for data...',
                                 ha='center', va='center', transform=self.ax_diff.transAxes)
                self.ax_diff.set_title('Differential Echo Profile')
            
            self.ax_diff.set_xlabel('Frame')
            self.ax_diff.set_ylabel('Distance Bin')
            
            plt.tight_layout()
            
            # Update display or save to file
            if self.interactive:
                try:
                    self.fig.canvas.draw()
                    self.fig.canvas.flush_events()
                except:
                    break
            else:
                # Save to file every 10 frames
                frame_counter += 1
                if frame_counter % 10 == 0:
                    filename = f'watchhand_frame_{frame_counter:06d}.png'
                    self.fig.savefig(filename, dpi=100, bbox_inches='tight')
                    print(f"Saved: {filename}")
            
            # Sleep to control update rate (10 FPS)
            time.sleep(0.1)
        
        print("Visualization stopped")
    
    def stop(self):
        """Stop the server."""
        self.running = False
        if self.client_socket:
            self.client_socket.close()
        if self.server_socket:
            self.server_socket.close()
        print("Server stopped")


def main():
    parser = argparse.ArgumentParser(description='WatchHand TCP Server')
    parser.add_argument('--host', default='0.0.0.0', help='Server host (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=9999, help='Server port (default: 9999)')
    args = parser.parse_args()
    
    server = WatchHandServer(host=args.host, port=args.port)
    try:
        server.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.stop()


if __name__ == '__main__':
    main()
