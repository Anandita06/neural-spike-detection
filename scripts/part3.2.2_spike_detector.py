import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==============================================================================
# CORE DETECTION FUNCTIONS
# ==============================================================================

def preprocess_signal(raw_data, window_size=100):
    """
    Removes slow baseline drift using a 100 ms (100 sample) moving median filter.
    Returns: detrended signal, baseline trend, and estimated noise sigma (via MAD).
    """
    padded_data = np.pad(raw_data, (window_size // 2, window_size // 2), mode='edge')
    moving_median = np.median(
        np.lib.stride_tricks.sliding_window_view(padded_data, window_size), axis=1
    )[:len(raw_data)]

    detrended = raw_data - moving_median

    # Robust Noise Floor Estimation via Median Absolute Deviation (MAD)
    mad = np.median(np.abs(detrended - np.median(detrended)))
    sigma_noise = mad / 0.6745

    return detrended, moving_median, sigma_noise


def detect_spikes_custom(signal, threshold, lockout=10):
    """
    Custom spike detector without external library dependencies.
    Enforces a refractory lockout period (lockout=10 samples = 10 ms)
    to capture rapid doublets while suppressing single-spike decay tails.
    """
    n = len(signal)
    detected_indices = []
    i = 0
    while i < n:
        if signal[i] > threshold:
            # Look across the lockout window for the local peak
            window_end = min(i + lockout, n)
            local_peak_offset = np.argmax(signal[i:window_end])
            true_peak = i + local_peak_offset

            detected_indices.append(true_peak)
            i += lockout  # Advance pointer past lockout window
        else:
            i += 1

    return np.array(detected_indices, dtype=int)


# ==============================================================================
# MAIN EXECUTION & CORRECTION VERIFICATION
# ==============================================================================

if __name__ == "__main__":
    # 1. Load Data
    project_dir = Path(__file__).resolve().parents[1]
    data = np.load(project_dir / 'data' / 'spike_data.npz')
    t = data['t']
    raw_data = data['raw_data']

    dt = t[1] - t[0]               # 0.001 s (1 ms)
    total_duration = t[-1] - t[0]  # 10.0 s

    # 2. Preprocess & Detrend
    detrended, moving_median, sigma_noise = preprocess_signal(raw_data, window_size=100)

    # 3. Dynamic Thresholding (4.0 * sigma_noise entering the stability plateau)
    multiplier = 4.0
    v_th = multiplier * sigma_noise
    lockout_samples = 10  # 10 ms lockout clears tails & captures fast doublets

    # 4. Detect Spikes
    spike_indices = detect_spikes_custom(detrended, v_th, lockout=lockout_samples)

    # Filter edge margin artifacts (first and last 20 samples)
    edge_margin = 20
    spike_indices = spike_indices[(spike_indices >= edge_margin) & (spike_indices <= len(detrended) - edge_margin)]

    detected_times = t[spike_indices]
    detected_voltages = raw_data[spike_indices]

    total_spikes = len(spike_indices)
    firing_rate = total_spikes / total_duration

    print("=" * 65)
    print("        PART 3: CORRECTED SPIKE DETECTION RESULTS")
    print("=" * 65)
    print(f"Estimated Noise Sigma (MAD):    {sigma_noise:.4f} V")
    print(f"Applied Voltage Threshold V_th: {v_th:.4f} V ({multiplier:.1f} sigma)")
    print(f"Total Detected Spikes:          {total_spikes}")
    print(f"Average Firing Rate:            {firing_rate:.2f} Hz")
    print("-" * 65)

    # ==========================================================================
    # METRIC CALCULATIONS: ISI & SPIKE DURATIONS
    # ==========================================================================
    # Inter-Spike Intervals (ISI)
    isi_ms = np.diff(detected_times) * 1000.0
    min_isi = np.min(isi_ms)
    max_isi = np.max(isi_ms)
    mean_isi = np.mean(isi_ms)

    # Spike Duration (Time above V_th)
    durations_ms = []
    for p_idx in spike_indices:
        idx_start = p_idx
        while idx_start > 0 and detrended[idx_start] > v_th:
            idx_start -= 1
        idx_end = p_idx
        while idx_end < len(detrended) - 1 and detrended[idx_end] > v_th:
            idx_end += 1
        durations_ms.append((t[idx_end] - t[idx_start]) * 1000.0)

    durations_ms = np.array(durations_ms)

    print("1. INTER-SPIKE INTERVALS (ISI):")
    print(f"   - Closest Distance (Min ISI):  {min_isi:.2f} ms")
    print(f"   - Farthest Distance (Max ISI): {max_isi:.2f} ms ({max_isi/1000.0:.3f} s)")
    print(f"   - Average Distance (Mean ISI): {mean_isi:.2f} ms")
    print("-" * 65)
    print("2. SPIKE DURATIONS (Time Above V_th):")
    print(f"   - Shortest Duration:           {np.min(durations_ms):.2f} ms")
    print(f"   - Longest Duration:            {np.max(durations_ms):.2f} ms")
    print(f"   - Average Duration:            {np.mean(durations_ms):.2f} ms")
    print("=" * 65)

    # ==========================================================================
    # PLOTTING DETECTED EVENTS
    # ==========================================================================
    plt.figure(figsize=(12, 5))
    plt.plot(t, raw_data, color='navy', linewidth=0.8, label='Raw Signal')
    plt.plot(t, moving_median, color='orange', linewidth=1.5, label='Extracted Baseline Drift')
    plt.plot(t, moving_median + v_th, color='red', linestyle='--', linewidth=1.0, alpha=0.7, label=f'Threshold ({v_th:.2f} V)')
    plt.plot(detected_times, detected_voltages, 'r.', markersize=8, label=f'Detected Events (N={total_spikes})')

    plt.title(f'Corrected Spike Detection Trace - Firing Rate: {firing_rate:.2f} Hz (N={total_spikes})', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)', fontsize=11)
    plt.ylabel('Voltage (V)', fontsize=11)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    output_path = project_dir / 'plots' / '03_spike_detection_trace.png'
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"\nPlot saved successfully as '{output_path}'!")