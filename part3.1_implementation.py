import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1. Load data
data = np.load('spike_data.npz')
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

# 3. Calculate Noise Floor and Set Threshold (4 * sigma)
mad = np.median(np.abs(detrended_data - np.median(detrended_data)))
sigma_noise = mad / 0.6745
v_th = 4.0 * sigma_noise  # ~1.23 V

# 4. Corrected Peak Detection Logic
lockout_samples = 5  # 5 ms window to clear the entire spike decay
edge_margin = 20
n_samples = len(detrended_data)

peak_indices = []
i = edge_margin

while i < (n_samples - edge_margin):
    # Check if current sample exceeds threshold
    if detrended_data[i] > v_th:
        # Define a local window around the crossing to find the true maximum peak
        search_window = detrended_data[i : i + lockout_samples]
        local_max_offset = np.argmax(search_window)
        true_peak_idx = i + local_max_offset
        
        peak_indices.append(true_peak_idx)
        
        # Jump index forward by lockout length so we don't re-trigger on the same spike
        i += lockout_samples
    else:
        i += 1

peak_indices = np.array(peak_indices)
detected_times = t[peak_indices]
detected_voltages = raw_data[peak_indices]

# 5. Compute Firing Rate
total_spikes = len(peak_indices)
firing_rate = total_spikes / total_duration

print("--- CORRECTED SPIKE DETECTION RESULTS ---")
print(f"Total Spikes Detected:   {total_spikes}")
print(f"Recording Duration:      {total_duration:.2f} s")
print(f"Average Firing Rate:     {firing_rate:.2f} Hz")

# 6. Final Validation Plot
plt.figure(figsize=(12, 5))
plt.plot(t, raw_data, color='navy', linewidth=0.8, label='Raw Data')
plt.plot(t, moving_median, color='orange', linewidth=1.5, label='Extracted Baseline Drift')
plt.plot(detected_times, detected_voltages, 'r.', markersize=7, label=f'Detected Spikes (N={total_spikes})')

plt.title(f'Spike Detection Result - Average Firing Rate: {firing_rate:.2f} Hz', fontsize=12, fontweight='bold')
plt.xlabel('Time (s)', fontsize=11)
plt.ylabel('Voltage (V)', fontsize=11)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('spike_detection_final_validation_fixed.png', dpi=300)
plt.close()

print("\nFixed plot saved as 'spike_detection_final_validation_fixed.png'!")