import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Load project directories dynamically
project_dir = Path(__file__).resolve().parents[1]
data_path = project_dir / 'data' / 'spike_data.npz'
plot_dir = project_dir / 'plots'
plot_dir.mkdir(parents=True, exist_ok=True)

# 1. Load Dataset
data = np.load(data_path)
t = data['t']
raw_data = data['raw_data']

total_duration = t[-1] - t[0]  # ~10.0 s

# 2. Preprocessing & Detrending (100 ms sliding median)
window_size = 100
padded_data = np.pad(raw_data, (window_size // 2, window_size // 2), mode='edge')
moving_median = np.median(
    np.lib.stride_tricks.sliding_window_view(padded_data, window_size), axis=1
)[:len(raw_data)]

detrended_data = raw_data - moving_median

# Estimate Noise Floor via Robust MAD
mad = np.median(np.abs(detrended_data - np.median(detrended_data)))
sigma_noise = mad / 0.6745

# 3. Spike Detection Function
def detect_spikes_loop(signal, threshold, lockout=10):
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

# 4. Sensitivity Sweep Across Threshold Multipliers
multipliers = [3.0, 3.5, 4.0, 4.5, 5.0]
lockout_samples = 10  # 10 ms lockout

print("=" * 65)
print("     PART 5: PARAMETER SENSITIVITY ANALYSIS")
print("=" * 65)
print(f"Noise Sigma (MAD): {sigma_noise:.4f} V | Duration: {total_duration:.2f} s")
print("-" * 65)
print(f"{'Multiplier':<12}{'Threshold (V)':<16}{'Spikes (N)':<14}{'Firing Rate (Hz)':<16}")
print("-" * 65)

results = []
for mult in multipliers:
    v_th = mult * sigma_noise
    spikes = detect_spikes_loop(detrended_data, v_th, lockout=lockout_samples)

    # Filter edge margin artifacts
    edge_margin = 20
    spikes = spikes[(spikes >= edge_margin) & (spikes <= len(detrended_data) - edge_margin)]

    rate = len(spikes) / total_duration
    results.append((mult, v_th, len(spikes), rate))

    label = " (Primary)" if mult == 4.0 else ""
    print(f"{mult:<12.1f}{v_th:<16.4f}{len(spikes):<14}{rate:<16.2f}{label}")

print("=" * 65)

# 5. Extract Primary Estimate (4.0 * sigma)
primary_mult = 4.0
v_th_primary = primary_mult * sigma_noise
primary_indices = detect_spikes_loop(detrended_data, v_th_primary, lockout=lockout_samples)
primary_indices = primary_indices[(primary_indices >= 20) & (primary_indices <= len(detrended_data) - 20)]

primary_spikes = len(primary_indices)
primary_rate = primary_spikes / total_duration
detected_times = t[primary_indices]
detected_voltages = raw_data[primary_indices]

# 6. Plot Primary Trace & Save
plt.figure(figsize=(12, 5))
plt.plot(t, raw_data, color='navy', linewidth=0.8, label='Raw Signal')
plt.plot(t, moving_median, color='orange', linewidth=1.5, label='Extracted Baseline Drift')
plt.plot(t, moving_median + v_th_primary, color='red', linestyle='--', linewidth=1.0, alpha=0.7, label=f'Threshold ({v_th_primary:.2f} V)')
plt.plot(detected_times, detected_voltages, 'r.', markersize=8, label=f'Primary Events (N={primary_spikes})')

plt.title(f'Part 5: Primary Spike Trace - Firing Rate: {primary_rate:.2f} Hz (N={primary_spikes})', fontsize=12, fontweight='bold')
plt.xlabel('Time (s)', fontsize=11)
plt.ylabel('Voltage (V)', fontsize=11)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

plot_out = plot_dir / 'part5_primary_detection_trace.png'
plt.savefig(plot_out, dpi=300)
plt.close()

print(f"\nPrimary detection trace saved to '{plot_out}'!")