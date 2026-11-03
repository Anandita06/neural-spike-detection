import time
import numpy as np
from pathlib import Path

# Load project directories dynamically
project_dir = Path(__file__).resolve().parents[1]
data_path = project_dir / 'data' / 'spike_data.npz'

# 1. Load Dataset & Preprocess
data = np.load(data_path)
t = data['t']
raw_data = data['raw_data']

window_size = 100
padded_data = np.pad(raw_data, (window_size // 2, window_size // 2), mode='edge')
moving_median = np.median(
    np.lib.stride_tricks.sliding_window_view(padded_data, window_size), axis=1
)[:len(raw_data)]

detrended_data = raw_data - moving_median

# Calculate 4.0 * sigma threshold via MAD
mad = np.median(np.abs(detrended_data - np.median(detrended_data)))
sigma_noise = mad / 0.6745
v_th = 4.0 * sigma_noise
lockout_samples = 10


# ==============================================================================
# VERSION A: REFERENCE IMPLEMENTATION (Python Loop)
# ==============================================================================
def detect_spikes_loop(signal, threshold, lockout=10):
    """
    Version A: Straightforward reference implementation using a Python loop.
    """
    n = len(signal)
    detected_indices = []
    i = 0
    while i < n:
        if signal[i] > threshold:
            window_end = min(i + lockout, n)
            local_peak_offset = np.argmax(signal[i:window_end])
            true_peak = i + local_peak_offset
            detected_indices.append(true_peak)
            i += lockout
        else:
            i += 1
    return np.array(detected_indices, dtype=int)


# ==============================================================================
# VERSION B: NUMPY VECTORIZED IMPLEMENTATION (No Sample-by-Sample Loop)
# ==============================================================================
def detect_spikes_vectorized(signal, threshold, lockout=10):
    """
    Version B: Vectorized NumPy implementation using boolean shifts to locate rising edges.
    """
    above_th = signal > threshold
    if not np.any(above_th):
        return np.array([], dtype=int)

    # Rising edge detection: True preceded by False
    rising_edges = np.flatnonzero(above_th[1:] & ~above_th[:-1]) + 1
    if above_th[0]:
        rising_edges = np.insert(rising_edges, 0, 0)

    if len(rising_edges) == 0:
        return np.array([], dtype=int)

    # Filter consecutive rising edges that fall within the lockout period
    filtered_events = []
    last_event = -lockout
    for edge in rising_edges:
        if edge - last_event >= lockout:
            window_end = min(edge + lockout, len(signal))
            peak_idx = edge + np.argmax(signal[edge:window_end])
            filtered_events.append(peak_idx)
            last_event = edge

    return np.array(filtered_events, dtype=int)


# ==============================================================================
# MAIN BENCHMARK & CORRECTNESS EVALUATION
# ==============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("      PART 6: NUMPY BENCHMARKING & CORRECTNESS EVALUATION")
    print("=" * 65)

    # 1. Evaluation 1: Correctness Check
    indices_loop = detect_spikes_loop(detrended_data, v_th, lockout_samples)
    indices_vec = detect_spikes_vectorized(detrended_data, v_th, lockout_samples)

    # Filter edge margin artifacts
    edge_margin = 20
    indices_loop = indices_loop[(indices_loop >= edge_margin) & (indices_loop <= len(detrended_data) - edge_margin)]
    indices_vec = indices_vec[(indices_vec >= edge_margin) & (indices_vec <= len(detrended_data) - edge_margin)]

    are_identical = np.array_equal(indices_loop, indices_vec)

    print("\nEVALUATION 1: CORRECTNESS CHECK")
    print("-" * 65)
    print(f"Version A (Loop) Event Count:        {len(indices_loop)}")
    print(f"Version B (Vectorized) Event Count:  {len(indices_vec)}")
    print(f"Outputs Exactly Identical?           {are_identical}")
    print("-" * 65)

    # 2. Evaluation 2: Performance Benchmark
    runs = 200
    times_loop = []
    times_vec = []

    for _ in range(runs):
        t0 = time.perf_counter()
        _ = detect_spikes_loop(detrended_data, v_th, lockout_samples)
        times_loop.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        _ = detect_spikes_vectorized(detrended_data, v_th, lockout_samples)
        times_vec.append(time.perf_counter() - t0)

    med_loop_ms = np.median(times_loop) * 1000.0
    med_vec_ms = np.median(times_vec) * 1000.0
    speedup = med_loop_ms / med_vec_ms

    print("\nEVALUATION 2: RUNTIME BENCHMARK RESULTS")
    print("-" * 65)
    print(f"{'Input Size (Samples)':<24}{'Timing Method':<20}{'Median Time (ms)':<18}")
    print("-" * 65)
    print(f"{len(raw_data):<24}{'perf_counter (Loop)':<20}{med_loop_ms:<18.3f}")
    print(f"{len(raw_data):<24}{'perf_counter (NumPy)':<20}{med_vec_ms:<18.3f}")
    print("-" * 65)
    print(f"MEASURED SPEEDUP FACTOR: {speedup:.2f}x Faster")
    print("=" * 65)