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

# 2. Detrending using 100 ms moving median
window_size = 100
padded_data = np.pad(raw_data, (window_size // 2, window_size // 2), mode='edge')
moving_median = np.median(np.lib.stride_tricks.sliding_window_view(padded_data, window_size), axis=1)[:len(raw_data)]
detrended_data = raw_data - moving_median

# 3. Calculate Noise Floor and Threshold (4 * sigma)
mad = np.median(np.abs(detrended_data - np.median(detrended_data)))
sigma_noise = mad / 0.6745
v_th = 4.0 * sigma_noise  # ~1.23 V

# 4. Extract Peaks with distance=15
peak_indices, _ = find_peaks(detrended_data, height=v_th, distance=15)
edge_margin = 20
valid_mask = (peak_indices >= edge_margin) & (peak_indices <= len(detrended_data) - edge_margin)
peak_indices = peak_indices[valid_mask]

detected_times = t[peak_indices]
detected_voltages = raw_data[peak_indices]

# Define requested verification intervals
intervals = [
    (0.0, 1.0),
    (1.5, 3.7),
    (4.3, 6.0),
    (6.3, 8.0),
    (8.4, 10.0)
]

# 5. Create 5-Subplot Verification Figure
fig, axes = plt.subplots(5, 1, figsize=(14, 15), sharey=True)

for idx, (t_start, t_end) in enumerate(intervals):
    mask = (t >= t_start) & (t <= t_end)

    # Plot raw signal & baseline offset
    axes[idx].plot(t[mask], raw_data[mask], color='navy', linewidth=0.9, label='Raw Data')
    axes[idx].plot(t[mask], moving_median[mask], color='orange', linewidth=1.2, label='Baseline Drift')
    axes[idx].plot(t[mask], moving_median[mask] + v_th, color='red', linestyle='--', linewidth=1.0, alpha=0.7, label='Threshold Line (V_th + Baseline)')

    # Plot detected spikes inside this window
    spikes_in_window = (detected_times >= t_start) & (detected_times <= t_end)
    window_spike_times = detected_times[spikes_in_window]
    window_spike_voltages = detected_voltages[spikes_in_window]
    window_spike_nums = np.where(spikes_in_window)[0] + 1

    axes[idx].plot(window_spike_times, window_spike_voltages, 'r.', markersize=9, label='Detected Spike')

    # Label each red dot with its Spike Number
    for stime, svolt, snum in zip(window_spike_times, window_spike_voltages, window_spike_nums):
        axes[idx].annotate(f"#{snum}", (stime, svolt + 0.2), color='red', fontweight='bold', fontsize=9, ha='center')

    axes[idx].set_title(f'Verification Window {idx+1}: t = {t_start}s to {t_end}s', fontsize=11, fontweight='bold')
    axes[idx].set_xlabel('Time (s)', fontsize=10)
    axes[idx].set_ylabel('Voltage (V)', fontsize=10)
    axes[idx].grid(True, linestyle='--', alpha=0.5)

axes[0].legend(loc='upper right', fontsize=9)
plt.tight_layout()
plt.savefig('spike_verification_closeups.png', dpi=300)
plt.close()

print("Close-up verification figure saved as 'spike_verification_closeups.png'!")