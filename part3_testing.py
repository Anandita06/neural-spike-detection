import numpy as np
import matplotlib
matplotlib.use('Agg')  # Prevents GUI renderer crashes
import matplotlib.pyplot as plt

# 1. Load the recording
data = np.load('spike_data.npz')
t = data['t']
raw_data = data['raw_data']

# 2. Remove baseline drift (100 ms sliding median window)
window_size = 100  # 100 samples = 100 ms at fs = 1000 Hz
padded_data = np.pad(raw_data, (window_size // 2, window_size // 2), mode='edge')
moving_median = np.median(np.lib.stride_tricks.sliding_window_view(padded_data, window_size), axis=1)[:len(raw_data)]

detrended_data = raw_data - moving_median

# 3. Calculate robust noise standard deviation (MAD)
mad = np.median(np.abs(detrended_data - np.median(detrended_data)))
sigma_noise = mad / 0.6745

v_th_3s = 3 * sigma_noise
v_th_4s = 4 * sigma_noise
v_th_5s = 5 * sigma_noise

print(f"Noise Standard Deviation (sigma): {sigma_noise:.4f} V")
print(f"3 * sigma threshold:              {v_th_3s:.4f} V")
print(f"4 * sigma threshold:              {v_th_4s:.4f} V")
print(f"5 * sigma threshold:              {v_th_5s:.4f} V")

# 4. Plot Detrended Voltage Histogram
plt.figure(figsize=(10, 5))
plt.hist(detrended_data, bins=150, density=True, color='skyblue', edgecolor='black', alpha=0.7, label='Detrended Voltage')

# Vertical threshold lines
plt.axvline(v_th_3s, color='orange', linestyle='--', linewidth=2, label=f'3σ ({v_th_3s:.2f} V)')
plt.axvline(v_th_4s, color='red', linestyle='--', linewidth=2, label=f'4σ ({v_th_4s:.2f} V)')
plt.axvline(v_th_5s, color='purple', linestyle='--', linewidth=2, label=f'5σ ({v_th_5s:.2f} V)')

plt.title('Histogram of Detrended Voltage & Spike Threshold Selection', fontsize=12, fontweight='bold')
plt.xlabel('Detrended Voltage (V)', fontsize=11)
plt.ylabel('Probability Density (Log Scale)', fontsize=11)
plt.yscale('log')  # Log scale highlights the rare spike tail on the right
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('histogram_threshold_analysis.png', dpi=300)
plt.close()

print("\nHistogram plot saved successfully as 'histogram_threshold_analysis.png'!")