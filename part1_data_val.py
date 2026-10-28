import numpy as np

# 1. Load the dataset
data = np.load('spike_data.npz')

# Print available keys/array names in the .npz archive
print("Keys in archive:", data.files)

# Extract arrays
t = data['t']
raw_data = data['raw_data']

# 2. Extract Data Audit metrics
num_samples_t = len(t)
num_samples_raw = len(raw_data)

shape_t = t.shape
shape_raw = raw_data.shape

dtype_t = t.dtype
dtype_raw = raw_data.dtype

# Calculate sampling parameters (assuming uniform time steps)
dt = t[1] - t[0]               # Sampling interval (seconds)
fs = 1.0 / dt                   # Sampling frequency (Hz)
time_span = t[-1] - t[0]        # Time between the first and last samples
nominal_duration = num_samples_t * dt  # Sample count divided by sampling rate

# 3. Print the Audit Summary
print("\n--- DATA AUDIT RESULTS ---")
print(f"Number of samples (t):       {num_samples_t}")
print(f"Number of samples (raw):     {num_samples_raw}")
print(f"Array shape (t):             {shape_t}")
print(f"Array shape (raw_data):      {shape_raw}")
print(f"Data type (t):               {dtype_t}")
print(f"Data type (raw_data):        {dtype_raw}")
print(f"Sampling interval (dt):      {dt:.6f} s ({dt*1e6:.2f} µs)")
print(f"Inferred sampling rate (fs):  {fs:.2f} Hz ({fs/1000:.2f} kHz)")
print(f"First-to-last sample span:    {time_span:.4f} s")
print(f"Nominal recording duration:   {nominal_duration:.4f} s")