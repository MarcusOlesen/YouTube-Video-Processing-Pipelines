"""
Audio Feature Extraction Test Suite (Sanity Check)

This file contains unit tests for the audio feature extraction pipeline. It tests various 
components of the audio analysis system using synthetic test signals:

- Sine waves (pure tones)
- Silence (zero signal)
- White noise (random signal)

Key test areas:
1. MFCC calculation
2. Audio feature extraction for different signal types
3. Volatility calculations
4. Feature aggregation across segments

Each test verifies expected behaviors like:
- Frequency detection for pure tones
- Energy distribution in frequency bands
- Silence detection
- Statistical aggregations

"""

import pytest
import numpy as np
import librosa
import subprocess

# ------------------- Tested fuctions -------------------
def extract_audio_from_video(video_path, audio_path):
    command = ['ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', audio_path, '-y']
    subprocess.run(command, check=True)

def calculate_mfcc(y, sr, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfccs


def analyze_audio_features(segment, sr):
    """
    Analyze audio features for a given audio segment.
    """
    # Calculate MFCCs and take the mean across time frames
    mfccs = calculate_mfcc(segment, sr)
    mean_mfccs = np.mean(mfccs, axis=1)
    
    # Calculate the frequency component of the volume contour around 4 Hz
    hop_length = 512
    rms = librosa.feature.rms(y=segment, hop_length=hop_length)[0]
    times = librosa.times_like(rms, sr=sr, hop_length=hop_length)
    rms_fft = np.fft.fft(rms)
    freqs = np.fft.fftfreq(len(rms), d=times[1] - times[0])
    idx_4hz = np.argmin(np.abs(freqs - 4.0))
    vol_contour_4hz = np.abs(rms_fft[idx_4hz])
    
    # Calculate the frequency centroid
    centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)[0]
    freq_centroid = np.mean(centroid)
    
    # Calculate the frequency bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr)[0]
    freq_bandwidth = np.mean(bandwidth)
    
    # Calculate the energy ratio in the range [0–630) Hz
    stft = np.abs(librosa.stft(y=segment))
    freqs = librosa.fft_frequencies(sr=sr)
    total_energy = np.sum(stft)
    low_energy = np.sum(stft[freqs < 630])
    energy_ratio_0_630 = low_energy / total_energy if total_energy > 0 else 0

    # Calculate the energy ratio in the range [630-1720) Hz
    mid_energy = np.sum(stft[(freqs >= 630) & (freqs < 1720)])
    energy_ratio_630_1720 = mid_energy / total_energy if total_energy > 0 else 0
    
    # Calculate the energy ratio in the range [1720-4400) Hz
    high_energy = np.sum(stft[(freqs >= 1720) & (freqs < 4400)])
    energy_ratio_1720_4400 = high_energy / total_energy if total_energy > 0 else 0
    
    # Calculate non-silence ratio
    amplitude_threshold = 0.02  # Amplitude threshold for silence
    non_silence = np.sum(np.abs(segment) > amplitude_threshold)
    non_silence_ratio = non_silence / len(segment)
    
    # Calculate volume dynamic range
    volume_dynamic_range = np.max(rms) - np.min(rms)
    
    # Calculate zero crossing rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(segment)[0]
    mean_zero_crossing_rate = np.mean(zero_crossing_rate)
    
    # Calculate volume standard deviation
    volume_std_dev = np.std(rms)
    
    return {
        "mean_mfccs": mean_mfccs,
        "vol_contour_4hz": vol_contour_4hz,
        "freq_centroid": freq_centroid,
        "freq_bandwidth": freq_bandwidth,
        "energy_ratio_0_630": energy_ratio_0_630,
        "energy_ratio_630_1720": energy_ratio_630_1720,
        "energy_ratio_1720_4400": energy_ratio_1720_4400,
        "non_silence_ratio": non_silence_ratio,
        "volume_dynamic_range": volume_dynamic_range,
        "mean_zero_crossing_rate": mean_zero_crossing_rate,
        "volume_std_dev": volume_std_dev
    }

def analyze_audio_segments(video_path, segment_length=10):
    # Extract the audio from the video using ffmpeg
    audio_path = "temp_audio.wav"
    extract_audio_from_video(video_path, audio_path)
    
    # Load the audio file
    y, sr = librosa.load(audio_path)
    
    # Determine the number of samples in each segment
    segment_samples = int(segment_length * sr)
    
    # Check if the audio is shorter than one segment
    if len(y) < segment_samples:
        # Handle this case by processing the entire audio as one segment
        segments = [y]
    else:
        # Frame the audio into segments
        segments = librosa.util.frame(y, frame_length=segment_samples, hop_length=segment_samples).T
        
    all_features = []
    for segment in segments:
        features = analyze_audio_features(segment, sr)
        all_features.append(features)
    
    return all_features


def calculate_volatility(values):
    differences = np.diff(values)
    volatility = np.std(differences) if not (len(differences) == 0 or np.all(differences == 0)) else np.nan
    return volatility

def aggregate_features(features_list):
    aggregated_features = {}
    # Initialize lists for MFCCs
    mfccs_list = []
    
    # Collect MFCCs separately
    for features in features_list:
        mfccs_list.append(features["mean_mfccs"])
    
    # Convert list of arrays to 2D array
    mfccs_array = np.array(mfccs_list)
    # Compute mean and std across segments for each MFCC coefficient
    aggregated_features["mean_mfccs"] = np.mean(mfccs_array, axis=0)
    aggregated_features["std_mfccs"] = np.std(mfccs_array, axis=0)
    # Compute volatility for MFCCs
    mfccs_diffs = np.diff(mfccs_array, axis=0)
    aggregated_features["volatility_mfccs"] = np.std(mfccs_diffs, axis=0)
    
    # Other scalar features
    scalar_keys = [key for key in features_list[0].keys() if key != "mean_mfccs"]
    for key in scalar_keys:
        all_values = [features[key] for features in features_list]
        mean_value = np.mean(all_values)
        std_value = np.std(all_values)
        volatility_value = calculate_volatility(all_values)
        aggregated_features[f"mean_{key}"] = mean_value
        aggregated_features[f"std_{key}"] = std_value
        aggregated_features[f"volatility_{key}"] = volatility_value
    
    return aggregated_features


def print_audio_features(features):
    formatted_output = "Audio Features:\n-------------------------\n"
    for key, value in features.items():
        if isinstance(value, np.ndarray):
            formatted_output += f"{key}: {value}\n"
        else:
            formatted_output += f"{key}: {value:.4f}\n"
    print(formatted_output)
    
# ------------------- Helper Function to Generate Test Signals -------------------

def generate_sine_wave(freq=440, sr=22050, duration=1.0):
    """Generate a sine wave of a given frequency and duration."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return 0.5 * np.sin(2 * np.pi * freq * t), sr

def generate_silence(sr=22050, duration=1.0):
    """Generate a silent audio signal."""
    return np.zeros(int(sr * duration)), sr

def generate_white_noise(sr=22050, duration=1.0):
    """Generate a white noise signal."""
    return np.random.uniform(-1, 1, int(sr * duration)), sr

# ------------------- Test Cases -------------------

def test_calculate_mfcc():
    """Test MFCC extraction with a sine wave."""
    y, sr = generate_sine_wave()
    mfccs = calculate_mfcc(y, sr)
    assert mfccs.shape[0] == 13  # Default n_mfcc=13
    assert mfccs.shape[1] > 0  # There should be time frames

def test_analyze_audio_features_silence():
    """Test feature extraction on silent audio."""
    y, sr = generate_silence()
    features = analyze_audio_features(y, sr)
    
    # Silence should have near-zero energy and a low non-silence ratio
    assert np.isclose(features["energy_ratio_0_630"], 0, atol=1e-5)
    assert np.isclose(features["energy_ratio_630_1720"], 0, atol=1e-5)
    assert np.isclose(features["energy_ratio_1720_4400"], 0, atol=1e-5)
    assert features["non_silence_ratio"] == 0

def test_analyze_audio_features_sine_wave():
    """Test feature extraction on a pure sine wave."""
    y, sr = generate_sine_wave(freq=440)
    features = analyze_audio_features(y, sr)

    # Check frequency centroid is close to 440 Hz
    assert np.isclose(features["freq_centroid"], 440, atol=50)

    # Check MFCC values (should not be NaN or Inf)
    assert not np.isnan(features["mean_mfccs"]).any()
    assert not np.isinf(features["mean_mfccs"]).any()

def test_analyze_audio_features_white_noise():
    """Test feature extraction on white noise."""
    y, sr = generate_white_noise()
    features = analyze_audio_features(y, sr)

    # Noise should have a high zero-crossing rate
    assert features["mean_zero_crossing_rate"] > 0.1

    # Noise should have a spread-out frequency spectrum
    assert features["freq_bandwidth"] > 1000

def test_calculate_volatility():
    """Test volatility calculation with synthetic values."""
    values = [1, 2, 3, 6, 10, 15]
    volatility = calculate_volatility(values)
    assert volatility > 0  # Should be positive

def test_aggregate_features():
    """Test feature aggregation across multiple segments."""
    feature_list = [
        {
            "mean_mfccs": np.array([1, 2, 3]),
            "freq_centroid": 1000,
            "freq_bandwidth": 500,
            "non_silence_ratio": 0.9,
        },
        {
            "mean_mfccs": np.array([2, 3, 4]),
            "freq_centroid": 1100,
            "freq_bandwidth": 600,
            "non_silence_ratio": 0.8,
        }
    ]
    aggregated = aggregate_features(feature_list)

    # Check mean calculations
    np.testing.assert_allclose(aggregated["mean_mfccs"], [1.5, 2.5, 3.5])
    assert np.isclose(aggregated["mean_freq_centroid"], 1050)
    assert np.isclose(aggregated["mean_freq_bandwidth"], 550)
    assert np.isclose(aggregated["mean_non_silence_ratio"], 0.85)

    # Check standard deviation calculations
    assert aggregated["std_non_silence_ratio"] > 0

    # Check volatility calculations
    assert aggregated["volatility_non_silence_ratio"] >= 0

if __name__ == "__main__":
    pytest.main()
