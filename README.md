# Neural Spike Detection

An end-to-end neural spike detection project for exploring noisy voltage
recordings, building and validating a transparent detector, estimating firing
rates, and benchmarking alternative implementations. It combines reproducible
analysis scripts, documented results, and visualizations of the detection
process.

**Author:** Anandita Banerjee

## Project layout

```text
data/
  spike_data.npz
plots/
  part3.1.0_histogram_threshold_analysis.png
  part3.1.1_spike_detection.png
  part3.2.2_spike_detection_trace.png
  part2_exploratory_plots.png
  part5_primary_detection_trace.png
scripts/
  part1_data_val.py
  part2_signal_explore.py
  part3.1.0_testing.py
  part3.1.1_implementation.py
  part3.2.2_spike_detector.py
  part4_unit_tests.py
  part5_sensitivity.py
  part6_numpy_benchmark.py
```

The data-analysis scripts locate the dataset and save generated plots relative
to the project directory, so they can be run from the repository root or
another working directory. The Part 4 unit tests do not require the dataset.

## Data

`data/spike_data.npz` contains two one-dimensional arrays:

| Property | Result |
| --- | ---: |
| Samples | 10,000 |
| Sampling interval | 0.001 s (1 ms) |
| Sampling frequency | 1,000 Hz (1 kHz) |
| First-to-last sample span | 9.999 s |
| Nominal duration (`samples / frequency`) | 10.000 s |
| Time array | `t`, shape `(10000,)`, `float64` |
| Recorded signal | `raw_data`, shape `(10000,)`, `float64` |

The signal summary from Part 2 is:

| Statistic | Voltage |
| --- | ---: |
| Minimum | -1.6910 |
| Maximum | 6.2020 |
| Mean | 0.0769 |
| Median | 0.0354 |
| Standard deviation | 0.7344 |

## Method and results

The detector removes slow baseline variation with a 100-sample moving median.
It estimates noise using the median absolute deviation (MAD), scaled as
`MAD / 0.6745`. The Part 3.2.2 detector applies a positive threshold of
`4 * sigma` to the detrended signal, groups nearby threshold crossings using
a 10-sample (10 ms) lockout, and excludes detections within 20 samples of
either recording edge.

For the included recording, the estimated noise sigma is 0.3066 V and the
4-sigma threshold is 1.2265 V. Part 3.2.2 detects 42 events. Dividing by the
recording duration gives an estimated firing rate of **4.20 Hz**. This is a
threshold-based estimate, not a ground-truth count.

Part 5 measures sensitivity to the threshold multiplier:

| Threshold | Threshold voltage | Detected events | Firing rate |
| ---: | ---: | ---: | ---: |
| 3.0 sigma | 0.9199 V | 50 | 5.00 Hz |
| 3.5 sigma | 1.0732 V | 44 | 4.40 Hz |
| 4.0 sigma | 1.2265 V | 42 | 4.20 Hz |
| 4.5 sigma | 1.3798 V | 42 | 4.20 Hz |
| 5.0 sigma | 1.5332 V | 42 | 4.20 Hz |

On this recording, the count is unchanged from 4 to 5 sigma, while lower
thresholds add detections. This suggests stability over that tested range,
but does not establish that every detected event is a biological spike.

Part 4 runs five synthetic tests covering no threshold crossings, an isolated
event, consecutive above-threshold samples, separate events, and a signal that
starts above threshold. All five pass. Part 6 compares loop and Boolean
threshold-crossing implementations: on this data they return identical event
indices (42 each). In a local run over 10,000 samples and 200 repetitions, the
median times were 0.903 ms for the reference loop and 0.098 ms for the NumPy
implementation, a measured speedup of 9.20x. Runtime varies by machine and
system load; agreement between implementations is not a substitute for
validation against labelled ground truth.

## Run the analysis

Use Python with NumPy and Matplotlib installed. SciPy is additionally required
by the Part 3.1.1 implementation:

```bash
python -m pip install numpy matplotlib scipy
```

From the repository root, run an individual assignment script as needed:

```bash
python scripts/part1_data_val.py
python scripts/part2_signal_explore.py
python scripts/part3.1.0_testing.py
python scripts/part3.2.2_spike_detector.py
python scripts/part4_unit_tests.py
python scripts/part5_sensitivity.py
python scripts/part6_numpy_benchmark.py
```

Part 3.1.0 compares 3-, 4-, and 5-sigma thresholds and saves a histogram.
Part 3.2.2 is the custom detector and prints its event, firing-rate, interval,
and duration summaries. Part 4 exits with an assertion error if a test fails.
Part 6 repeats each timing measurement 200 times and reports median runtimes.

## Figures

| Figure | Contents |
| --- | --- |
| [`part2_exploratory_plots.png`](plots/part2_exploratory_plots.png) | Full recording and two close-up signal views. |
| [`part3.1.0_histogram_threshold_analysis.png`](plots/part3.1.0_histogram_threshold_analysis.png) | Detrended-signal distribution with 3-, 4-, and 5-sigma thresholds. |
| [`part3.1.1_spike_detection.png`](plots/part3.1.1_spike_detection.png) | Detection trace from the Part 3.1.1 implementation. |
| [`part3.2.2_spike_detection_trace.png`](plots/part3.2.2_spike_detection_trace.png) | Part 3.2.2 detections, baseline, and 4-sigma threshold. |
| [`part5_primary_detection_trace.png`](plots/part5_primary_detection_trace.png) | Part 5 primary 4-sigma detection trace. |

The figures are committed outputs. Parts 2, 3.1.0, 3.1.1, 3.2.2, and 5
regenerate their corresponding plots. Part 3.2.2 also prints the first 10
detected spikes with their zero-based sample indices and timestamps in seconds.

## Interpretation and limitations

- The implementation identifies candidate spikes as positive excursions above
  a MAD-scaled threshold, grouping nearby crossings with a lockout interval.
  Accordingly, reported firing rates are method-dependent estimates rather
  than ground-truth spike labels.
- The firing-rate estimate assumes detected positive events correspond to
  spikes and that the analyzed recording duration is representative.
- A fixed threshold multiplier and lockout period may miss small or closely
  spaced events, or count noise as an event. Threshold sensitivity is evaluated
  only over the multipliers listed above.
- The 20-sample edge exclusion removes detections near the start and end.
- The five Part 4 tests check selected synthetic cases; they do not measure
  accuracy on labelled recordings.
- Part 6 checks that two implementations agree on this input and compares
  runtime. Neither result proves the detector is scientifically correct.
