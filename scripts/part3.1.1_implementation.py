import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# 1. Load data
project_dir = Path(__file__).resolve().parents[1]
data = np.load(project_dir / 'data' / 'spike_data.npz')
t = data['t']
raw_data = data['raw_data']

# Sampling parameters
dt = t[1] - t[0]               # 0.001 s (1 ms)
fs = 1.0 / dt                   # 1000 Hz
total_duration = t[-1] - t[0]   # ~10.0 s

# 2. Detrending using 100 ms moving median
window_size = 100
padded_data = np.pad(raw_data, (window_size // 2, window_size // 2), mode='edge')
moving_median = np.median(np.lib.stride_tricks.sliding_window_view(padded_data, window_size), axis=1)[:len(raw_data)]
detrended_data = raw_data - moving_median

# 3. Calculate Noise Floor and Threshold (4 * sigma)
mad = np.median(np.abs(detrended_data - np.median(detrended_data)))
sigma_noise = mad / 0.6745
v_th = 4.0 * sigma_noise  # ~1.23 V

# 4. Single-Peak Extraction using scipy.signal.find_peaks
# distance=15 enforces a 15 ms refractory period so tails are NEVER counted
peak_indices, _ = find_peaks(
    detrended_data,
    height=v_th,
    distance=15
)

# Filter out edge artifacts (first and last 20 ms)
edge_margin = 20
valid_mask = (peak_indices >= edge_margin) & (peak_indices <= len(detrended_data) - edge_margin)
peak_indices = peak_indices[valid_mask]

detected_times = t[peak_indices]
detected_voltages = raw_data[peak_indices]

# 5. Compute Firing Rate
total_spikes = len(peak_indices)
firing_rate = total_spikes / total_duration

print("--- FINAL CLEAN SPIKE DETECTION RESULTS ---")
print(f"Total Spikes Detected:   {total_spikes}")
print(f"Recording Duration:      {total_duration:.2f} s")
print(f"Average Firing Rate:     {firing_rate:.2f} Hz")

# 6. Final Validation Plot
plt.figure(figsize=(12, 5))
plt.plot(t, raw_data, color='navy', linewidth=0.8, label='Raw Data')
plt.plot(t, moving_median, color='orange', linewidth=1.5, label='Extracted Baseline Drift')
plt.plot(detected_times, detected_voltages, 'r.', markersize=8, label=f'Detected Spikes (N={total_spikes})')

plt.title(f'Spike Detection Result - Average Firing Rate: {firing_rate:.2f} Hz', fontsize=12, fontweight='bold')
plt.xlabel('Time (s)', fontsize=11)
plt.ylabel('Voltage (V)', fontsize=11)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(project_dir / 'plots' / 'spike_detection.png', dpi=300)
plt.close()

print("\nSpike detection plot saved as 'spike_detection.png'!")