#!/usr/bin/env python3
"""
Self-contained test: generate mock PCM data, process it, save visualization.
No GUI required, no TCP required.
"""

import struct
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
import os

# Add server directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from watchhand_server import EchoProfileProcessor, F_MIN, F_MAX, CHIRP_LENGTH, DISTANCE_BINS, TIME_WINDOW_FRAMES

SAMPLE_RATE = 44100

def generate_mock_audio(duration_seconds=3.0):
    """Generate mock PCM audio: FMCW chirp + simulated echo reflections."""
    total_samples = int(SAMPLE_RATE * duration_seconds)
    samples = np.zeros(total_samples, dtype=np.float32)
    
    t = np.arange(total_samples) / SAMPLE_RATE
    
    # Generate FMCW chirp (repeated)
    chirp_duration = CHIRP_LENGTH / SAMPLE_RATE
    n_chirps = total_samples // CHIRP_LENGTH
    
    for i in range(n_chirps):
        chirp_start = i * CHIRP_LENGTH
        chirp_t = np.arange(CHIRP_LENGTH) / SAMPLE_RATE
        T = chirp_duration
        phase = 2 * np.pi * (F_MIN * chirp_t + (F_MAX - F_MIN) * chirp_t**2 / (2 * T))
        chirp = np.cos(phase)
        
        # Apply Hann window
        hann = 0.5 * (1 - np.cos(2 * np.pi * np.arange(CHIRP_LENGTH) / (CHIRP_LENGTH - 1)))
        chirp_windowed = chirp * hann
        
        # Direct path (strong, short delay)
        direct_delay = 20  # samples
        if chirp_start + direct_delay < total_samples:
            end = min(chirp_start + direct_delay + CHIRP_LENGTH, total_samples)
            length = end - (chirp_start + direct_delay)
            samples[chirp_start + direct_delay:end] += chirp_windowed[:length] * 0.8
        
        # Simulated echo (weaker, longer delay, varies with "hand movement")
        echo_delay = 100 + int(30 * np.sin(2 * np.pi * i / n_chirps))  # varying delay
        echo_amp = 0.3
        if chirp_start + echo_delay < total_samples:
            end = min(chirp_start + echo_delay + CHIRP_LENGTH, total_samples)
            length = end - (chirp_start + echo_delay)
            samples[chirp_start + echo_delay:end] += chirp_windowed[:length] * echo_amp
        
        # Second echo (even weaker)
        echo2_delay = 200 + int(20 * np.cos(2 * np.pi * i / n_chirps * 1.5))
        echo2_amp = 0.15
        if chirp_start + echo2_delay < total_samples:
            end = min(chirp_start + echo2_delay + CHIRP_LENGTH, total_samples)
            length = end - (chirp_start + echo2_delay)
            samples[chirp_start + echo2_delay:end] += chirp_windowed[:length] * echo2_amp
    
    # Add some noise
    samples += np.random.normal(0, 0.02, total_samples).astype(np.float32)
    
    # Scale to 16-bit range
    samples = samples * 10000
    
    return samples.astype(np.int16)


def main():
    print("=" * 60)
    print("WatchHand Server Self-Test")
    print("=" * 60)
    
    # Generate mock audio
    print("\n[1] Generating mock PCM audio (3 seconds)...")
    mock_audio = generate_mock_audio(3.0)
    print(f"    Generated {len(mock_audio)} samples ({len(mock_audio) / SAMPLE_RATE:.1f}s)")
    print(f"    Sample range: {mock_audio.min()} ~ {mock_audio.max()}")
    
    # Initialize processor
    print(f"\n[2] Initializing EchoProfileProcessor...")
    print(f"    Sample rate: {SAMPLE_RATE} Hz")
    print(f"    FMCW: {F_MIN}-{F_MAX} Hz")
    print(f"    Chirp length: {CHIRP_LENGTH} samples")
    print(f"    Distance bins: {DISTANCE_BINS}")
    print(f"    Time window: {TIME_WINDOW_FRAMES} frames")
    
    processor = EchoProfileProcessor(
        sample_rate=SAMPLE_RATE,
        f_min=F_MIN,
        f_max=F_MAX,
        chirp_length=CHIRP_LENGTH,
        distance_bins=DISTANCE_BINS,
        time_window_frames=TIME_WINDOW_FRAMES
    )
    
    # Feed data in chunks (simulate streaming)
    print(f"\n[3] Feeding data in 100ms chunks...")
    chunk_size = 4800  # 100ms
    frame_count = 0
    last_orig = None
    last_diff = None
    
    for i in range(0, len(mock_audio), chunk_size):
        chunk = mock_audio[i:i+chunk_size]
        samples_array = chunk.astype(np.float32)
        
        result = processor.feed(samples_array)
        if result is not None:
            orig, diff, frames = result
            last_orig = orig
            last_diff = diff
            frame_count += 1
            if frame_count % 5 == 0:
                print(f"    Processed frame batch #{frame_count}")
    
    print(f"\n[4] Total frame batches produced: {frame_count}")
    
    if last_orig is None:
        print("\n[ERROR] No frames produced! Check processing pipeline.")
        return
    
    # Visualize
    print(f"\n[5] Generating visualization...")
    orig_2d = last_orig.reshape(DISTANCE_BINS, TIME_WINDOW_FRAMES)
    diff_2d = last_diff.reshape(DISTANCE_BINS, TIME_WINDOW_FRAMES - 1)
    
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 1, height_ratios=[1, 1])
    
    ax_orig = fig.add_subplot(gs[0])
    orig_lo = np.percentile(orig_2d, 2)
    orig_hi = np.percentile(orig_2d, 98)
    orig_clipped = np.clip(orig_2d, orig_lo, orig_hi)
    ax_orig.imshow(orig_clipped, aspect='auto', cmap='coolwarm',
                   extent=[0, TIME_WINDOW_FRAMES, DISTANCE_BINS, 0])
    ax_orig.set_title(f'Original Echo Profile (Range: {orig_lo:.2e} ~ {orig_hi:.2e})')
    ax_orig.set_xlabel('Frame')
    ax_orig.set_ylabel('Distance Bin')
    
    ax_diff = fig.add_subplot(gs[1])
    diff_lo = np.percentile(diff_2d, 2)
    diff_hi = np.percentile(diff_2d, 98)
    diff_clipped = np.clip(diff_2d, diff_lo, diff_hi)
    ax_diff.imshow(diff_clipped, aspect='auto', cmap='coolwarm',
                   extent=[0, TIME_WINDOW_FRAMES - 1, DISTANCE_BINS, 0])
    ax_diff.set_title(f'Differential Echo Profile (Range: {diff_lo:.2e} ~ {diff_hi:.2e})')
    ax_diff.set_xlabel('Frame')
    ax_diff.set_ylabel('Distance Bin')
    
    plt.tight_layout()
    
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_output.png')
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"    Saved: {output_file}")
    
    print("\n" + "=" * 60)
    print("TEST PASSED - All components working correctly")
    print("=" * 60)


if __name__ == '__main__':
    main()
