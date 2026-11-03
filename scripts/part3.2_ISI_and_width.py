import numpy as np
import scipy.signal as signal
from pathlib import Path

# 1. Load data
project_dir = Path(__file__).resolve().parents[1]
data = np.load(project_dir / 'data' / 'spike_data.npz')
t = data['t']
raw_data = data['raw_data']

dt = t[1] - t[0]  # 0.001 s = 1.0 ms
fs = 1.0 / dt

# 2. Detrending using 100 ms moving median
window_size = 100
padded_data = np.pad(raw_data, (window_size // 2, window_size // 2), mode='edge')
moving_median = np.median(np.lib.stride_tricks.sliding_window_view(padded_data, window_size), axis=1)[:len(raw_data)]
detrended_data = raw_data - moving_median

# 3. Calculate Threshold (4 * sigma via MAD)
mad = np.median(np.abs(detrended_data - np.median(detrended_data)))
sigma_noise = mad / 0.6745
v_th = 4.0 * sigma_noise  # ~1.23 V

# 4. Extract Peak Indices
peak_indices, _ = signal.find_peaks(detrended_data, height=v_th, distance=15)
edge_margin = 20
valid_mask = (peak_indices >= edge_margin) & (peak_indices <= len(detrended_data) - edge_margin)
peak_indices = peak_indices[valid_mask]
detected_times = t[peak_indices]

# ==============================================================================
# CALCULATION 1: Inter-Spike Intervals (ISI)
# ==============================================================================
isi_seconds = np.diff(detected_times)
isi_ms = isi_seconds * 1000.0  # Convert to milliseconds

min_isi_ms = np.min(isi_ms)
max_isi_ms = np.max(isi_ms)
mean_isi_ms = np.mean(isi_ms)

min_isi_pair = (detected_times[np.argmin(isi_ms)], detected_times[np.argmin(isi_ms) + 1])
max_isi_pair = (detected_times[np.argmax(isi_ms)], detected_times[np.argmax(isi_ms) + 1])

# ==============================================================================
# CALCULATION 2: Spike Duration (Width Above Threshold)
# ==============================================================================
spike_durations_ms = []
start_times = []
end_times = []

for p_idx in peak_indices:
    # Walk backward from peak to find upward threshold crossing
    idx_start = p_idx
    while idx_start > 0 and detrended_data[idx_start] > v_th:
        idx_start -= 1

    # Walk forward from peak to find downward threshold crossing
    idx_end = p_idx
    while idx_end < len(detrended_data) - 1 and detrended_data[idx_end] > v_th:
        idx_end += 1

    duration_s = t[idx_end] - t[idx_start]
    duration_msec = duration_s * 1000.0

    spike_durations_ms.append(duration_msec)
    start_times.append(t[idx_start])
    end_times.append(t[idx_end])

spike_durations_ms = np.array(spike_durations_ms)

min_dur_ms = np.min(spike_durations_ms)
max_dur_ms = np.max(spike_durations_ms)
mean_dur_ms = np.mean(spike_durations_ms)

# ==============================================================================
# PRINT SUMMARY REPORT
# ==============================================================================
print("=" * 65)
print("             SPIKE METRICS ANALYSIS SUMMARY")
print("=" * 65)
print(f"Total Spikes Analyzed:          {len(peak_indices)}")
print("-" * 65)
print("1. INTER-SPIKE INTERVALS (Distance Between Spikes):")
print(f"   - Closest Distance (Min ISI): {min_isi_ms:.2f} ms  (between t={min_isi_pair[0]:.3f}s and t={min_isi_pair[1]:.3f}s)")
print(f"   - Farthest Distance (Max ISI):{max_isi_ms:.2f} ms ({max_isi_ms/1000.0:.3f} s) (between t={max_isi_pair[0]:.3f}s and t={max_isi_pair[1]:.3f}s)")
print(f"   - Mean Distance (Avg ISI):    {mean_isi_ms:.2f} ms")
print("-" * 65)
print("2. SPIKE DURATIONS (Time Above Threshold V_th):")
print(f"   - Shortest Spike Duration:   {min_dur_ms:.2f} ms")
print(f"   - Longest Spike Duration:    {max_dur_ms:.2f} ms")
print(f"   - Average Spike Duration:    {mean_dur_ms:.2f} ms")
print("=" * 65)

# Detailed per-spike summary table (first 10 spikes preview)
print("\nFirst 10 Spikes Breakdown:")
print(f"{'Spike #':<8}{'Peak Time (s)':<15}{'Cross Start (s)':<18}{'Cross End (s)':<16}{'Duration (ms)':<15}")
print("-" * 72)
for i in range(min(10, len(peak_indices))):
    print(f"{i+1:<8}{detected_times[i]:<15.3f}{start_times[i]:<18.3f}{end_times[i]:<16.3f}{spike_durations_ms[i]:<15.2f}")