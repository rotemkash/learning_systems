"""
Written and submitted by Rotem Kashani 209073352 and David Koplev 208870279
"""

import wave
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import numpy as np
from sklearn.decomposition import FastICA
import librosa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# List of file paths for the source audio files
file_paths = ["source1.wav", "source2.wav", "source3.wav",
              "source4.wav", "source5.wav", "source6.wav"]

# List to store the audio signals
signals = []

# Loop through each file path
for path in file_paths:
    # Open the WAV file
    with wave.open(path, 'rb') as wav_file:
        # Read the audio frames
        signal = wav_file.readframes(-1)
        # Convert the bytes to a NumPy array
        signal = np.frombuffer(signal, dtype=np.int16)
    # Append the signal to the signals list
    signals.append(signal)

# Plot the spectrogram for each audio signal
for i, signal in enumerate(signals):
    plt.figure(figsize=(10, 4))
    plt.title(f"Spectrogram of source {i+1}")
    # Compute the spectrogram
    f, t, Sxx = spectrogram(signal, fs=44100)
    # Plot the spectrogram
    plt.pcolormesh(t, f, Sxx)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

# Generate a random mixing matrix
mixing_matrix = np.random.uniform(0.5, 2.5, size=(6, 6))

# Mix signals using the mixingt matrix
mixed_signals = np.dot(mixing_matrix,np.array(signals))

# Save mixed signals to WAV files
for i,mixed_signal in enumerate(mixed_signals):
    mixed_signal = mixed_signal.astype(np.int16)
    with wave.open(f"mixed_source{i+1}.wav" ,'wb') as wav_file:
        # Mono audio
        wav_file.setnchannels(1)
        # 16-bit audio
        wav_file.setsampwidth(2)
        wav_file.setframerate(44100)
        wav_file.writeframes(mixed_signal.tobytes())

# Load mixed signals from WAV files
mixed_signals = []
for i in range(1, 7):
    with wave.open(f"mixed_source{i}.wav", 'rb') as wav_file:
        signal = wav_file.readframes(-1)
        signal = np.frombuffer(signal, dtype=np.int16)
        mixed_signals.append(signal)

# Convert mixed signals to numpy array
mixed_signals = np.array(mixed_signals)

# Apply ICA
ica = FastICA(n_components=6, random_state=42)
recovered_signals = ica.fit_transform(mixed_signals.T).T

# Normalize recovered signals
recovered_signals -= np.mean(recovered_signals, axis=1, keepdims=True)
recovered_signals /= np.sqrt(np.mean(recovered_signals ** 2, axis=1, keepdims=True))

# Save recovered signals to WAV files
for i, signal in enumerate(recovered_signals):
    signal = (signal * 32767).astype(np.int16)
    with wave.open(f"recovered_source{i+1}.wav", 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono audio
        wav_file.setsampwidth(2)  # 16-bit audio
        wav_file.setframerate(44100)
        wav_file.writeframes(signal.tobytes())
        
# Plot spectrogram for each recovered signal
for i, signal in enumerate(recovered_signals):
    plt.figure(figsize=(10, 4))
    plt.title(f"Spectrogram of recovered source {i+1}")
    f, t, Sxx = spectrogram(signal, fs=44100)
    plt.pcolormesh(t, f, Sxx)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

def extract_features(signal, sr):
    # Extract MFCCs (Mel-Frequency Cepstral Coefficients)
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)

    # Extract spectral centroid, bandwidth, zero crossing rate, and RMS energy
    centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(signal)
    rms_energy = librosa.feature.rms(y=signal)

    # Concatenate all features into a single array
    features = np.vstack([mfccs, centroid, bandwidth, zero_crossing_rate, rms_energy])

    # Pad or truncate features to a fixed length (e.g., 100 frames)
    max_length = 100
    if features.shape[1] < max_length:
        # Pad with zeros if the number of frames is less than max_length
        features = np.pad(features, ((0, 0), (0, max_length - features.shape[1])), mode='constant')
    elif features.shape[1] > max_length:
        # Truncate if the number of frames is greater than max_length
        features = features[:, :max_length]

    return features


# Define segment length in seconds
segment_length = 0.5

# Define hop length for segmentation
hop_length = int(44100 * segment_length)  # Assuming a sampling rate of 44100 Hz

# Initialize lists to store features and labels
all_features = []
all_labels = []

# Iterate over each original signal and segment it
for i, signal in enumerate(signals):
    # Calculate the number of segments
    num_segments = len(signal) // hop_length + 1 if len(signal) % hop_length != 0 else len(signal) // hop_length

    # Iterate over each segment
    for j in range(num_segments):
        # Check if the segment exceeds the length of the signal
        if (j + 1) * hop_length > len(signal):
            segment = signal[j * hop_length:]
        else:
            segment = signal[j * hop_length: (j + 1) * hop_length]

        # Normalize segment to floating-point format
        segment_float = librosa.util.normalize(segment.astype(np.float32))

        # Extract features
        features = extract_features(segment_float, 44100)

        # Append features to the list
        all_features.append(features)

        # Assign label based on the original signal
        label = i
        all_labels.append(label)

# Convert lists to numpy arrays
all_features = np.array(all_features)
all_labels = np.array(all_labels)



# Split data into training and test sets with stratification
X_train, X_test, y_train, y_test = train_test_split(all_features, all_labels, test_size=0.5, random_state=42, stratify=all_labels)

# Flatten the features array
X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)

# Initialize Random Forest classifier with adjusted parameters
clf = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=42)

# Train classifier
clf.fit(X_train_flattened, y_train)

# Predict labels for test set
y_pred = clf.predict(X_test_flattened)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)