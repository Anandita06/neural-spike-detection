import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Bypass Tkinter GUI and render directly to file
import matplotlib.pyplot as plt

# 1. Load data
project_dir = Path(__file__).resolve().parents[1]
data = np.load(project_dir / 'data' / 'spike_data.npz')
t = data['t']
raw_data = data['raw_data']

# 2. Numerical Summary
min_val = np.min(raw_data)
max_val = np.max(raw_data)
mean_val = np.mean(raw_data)
median_val = np.median(raw_data)
std_val = np.std(raw_data)

print("--- NUMERICAL SUMMARY OF SIGNAL ---")
print(f"Minimum:            {min_val:.4f}")
print(f"Maximum:            {max_val:.4f}")
print(f"Mean:               {mean_val:.4f}")
print(f"Median:             {median_val:.4f}")
print(f"Standard Deviation: {std_val:.4f}")

# 3. Visualization
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharey=True)

# Plot 1: Entire 10-second recording
axes[0].plot(t, raw_data, color='navy', linewidth=0.8)
axes[0].set_title('Full Recording (10 Seconds) - Overview of Baseline Drift and Noise', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Voltage (V)')
axes[0].grid(True, linestyle='--', alpha=0.6)

# Plot 2: Close-up Interval 1 (t = 4.5s to 6.0s)
t_start1, t_end1 = 4.5, 6.0
mask1 = (t >= t_start1) & (t <= t_end1)
axes[1].plot(t[mask1], raw_data[mask1], color='darkred', linewidth=1.2)
axes[1].set_title(f'Close-Up Interval 1 ({t_start1}s to {t_end1}s) - Highlighting Transient Spikes', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Voltage (V)')
axes[1].grid(True, linestyle='--', alpha=0.6)

# Plot 3: Close-up Interval 2 (t = 8.0s to 9.5s)
t_start2, t_end2 = 8.0, 9.5
mask2 = (t >= t_start2) & (t <= t_end2)
axes[2].plot(t[mask2], raw_data[mask2], color='darkgreen', linewidth=1.2)
axes[2].set_title(f'Close-Up Interval 2 ({t_start2}s to {t_end2}s) - Highlighting Baseline Drift / Noise Baseline', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('Voltage (V)')
axes[2].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig(project_dir / 'plots' / 'part2_exploratory_plots.png', dpi=300)
print("\nPlot saved successfully as 'part2_exploratory_plots.png'!")