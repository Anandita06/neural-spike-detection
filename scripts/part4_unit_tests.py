import numpy as np

# ==============================================================================
# CORE DETECTION FUNCTION (Reference Implementation for Testing)
# ==============================================================================

def detect_spikes_loop(signal, threshold, lockout=5):
    """
    Core spike detection function using a Python loop.
    Identifies threshold-crossing events and applies a refractory lockout period.
    """
    n = len(signal)
    detected_indices = []
    i = 0
    while i < n:
        if signal[i] > threshold:
            # Search within the lockout window for the local peak
            window_end = min(i + lockout, n)
            local_peak_offset = np.argmax(signal[i:window_end])
            true_peak = i + local_peak_offset

            detected_indices.append(true_peak)
            i += lockout  # Lockout period to prevent multi-counting
        else:
            i += 1

    return np.array(detected_indices, dtype=int)


# ==============================================================================
# PART 4: UNIT TESTING SUITE
# ==============================================================================

def run_unit_tests():
    print("=" * 65)
    print("         PART 4: RUNNING SPIKE DETECTOR UNIT TESTS")
    print("=" * 65)

    threshold = 1.0
    lockout = 5

    # --------------------------------------------------------------------------
    # Test 1: No samples above threshold -> Expected: 0 events
    # --------------------------------------------------------------------------
    test1_signal = np.array([0.1, 0.5, 0.2, -0.4, 0.8, 0.9, -0.1])
    res1 = detect_spikes_loop(test1_signal, threshold, lockout)
    assert len(res1) == 0, f"Test 1 Failed: Expected 0 events, got {len(res1)}"
    print("[PASS] Test 1: No samples above threshold -> Detected 0 events")

    # --------------------------------------------------------------------------
    # Test 2: One isolated sample above threshold -> Expected: 1 event
    # --------------------------------------------------------------------------
    test2_signal = np.array([0.1, 0.2, 1.5, 0.3, 0.1])
    res2 = detect_spikes_loop(test2_signal, threshold, lockout)
    assert len(res2) == 1 and res2[0] == 2, f"Test 2 Failed: Expected 1 event at index 2, got {res2}"
    print("[PASS] Test 2: One isolated sample above threshold -> Detected 1 event (at index 2)")

    # --------------------------------------------------------------------------
    # Test 3: Several consecutive samples above threshold -> Expected: 1 event (Peak)
    # --------------------------------------------------------------------------
    test3_signal = np.array([0.1, 1.2, 1.8, 1.4, 1.1, 0.2])
    res3 = detect_spikes_loop(test3_signal, threshold, lockout)
    assert len(res3) == 1 and res3[0] == 2, f"Test 3 Failed: Expected 1 event at peak index 2, got {res3}"
    print("[PASS] Test 3: Consecutive above-threshold samples -> Grouped into 1 event (at peak index 2)")

    # --------------------------------------------------------------------------
    # Test 4: Two separate above-threshold intervals -> Expected: 2 events
    # --------------------------------------------------------------------------
    # Intervals separated by more than lockout duration (5 samples)
    test4_signal = np.array([0.1, 1.5, 0.2, 0.1, 0.1, 0.2, 0.1, 1.8, 0.2])
    res4 = detect_spikes_loop(test4_signal, threshold, lockout)
    assert len(res4) == 2 and res4[0] == 1 and res4[1] == 7, f"Test 4 Failed: Expected 2 events at indices [1, 7], got {res4}"
    print("[PASS] Test 4: Separate above-threshold intervals -> Detected 2 events (indices 1 & 7)")

    # --------------------------------------------------------------------------
    # Test 5: Signal starts above threshold -> Expected: 1 event at start
    # --------------------------------------------------------------------------
    test5_signal = np.array([1.5, 1.2, 0.2, 0.1, 0.4])
    res5 = detect_spikes_loop(test5_signal, threshold, lockout)
    assert len(res5) == 1 and res5[0] == 0, f"Test 5 Failed: Expected 1 event at index 0, got {res5}"
    print("[PASS] Test 5: Signal starts above threshold -> Detected initial event (index 0)")

    print("=" * 65)
    print("ALL 5 UNIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    run_unit_tests()