from math import *
from statistics import *
from typing import Callable, Literal

import cv2
import matplotlib.pyplot as plt
import noisereduce as nr
import numpy as np
import pywt
import scipy.fft as fft
import soundfile as sf
from moviepy import AudioClip, AudioFileClip, VideoFileClip
from scipy import fft, signal
from scipy.interpolate import interp1d
from scipy.io import wavfile
from scipy.signal import find_peaks
from utils import *
from visualisations import *

def stereo_median_filter(
    audio_samples: list[list[float]], kernel_size: int = 3
) -> list[list[float]]:
    left_channel, right_channel = stereo_to_mono(audio_samples)
    filtered_left = signal.medfilt(left_channel, kernel_size)
    filtered_right = signal.medfilt(right_channel, kernel_size)
    return mono_to_stereo(filtered_left, filtered_right)


def stereo_wavelet_denoise(
    audio_samples: list[list[float]],
    threshold_factor: float = 1.0,
    wavelet="db1",
    level=None,
) -> list[list[float]]:
    left_channel, right_channel = stereo_to_mono(audio_samples)

    def wavelet_denoise(channel: list[float]) -> list[float]:
        coeffs = pywt.wavedec(channel, wavelet, level=level)
        sigma = median(abs(coeffs[-1])) / 0.6745
        uthresh = threshold_factor * sigma * sqrt(2 * log(len(channel)))
        denoised_coeffs = [
            pywt.threshold(c, value=uthresh, mode="soft") for c in coeffs
        ]
        return pywt.waverec(denoised_coeffs, wavelet)

    filtered_left = wavelet_denoise(left_channel)
    filtered_right = wavelet_denoise(right_channel)

    return mono_to_stereo(filtered_left, filtered_right)

def stereo_decrackle_filter(audio: list[list[float]], threshold_factor: float = 3.0, window_size: int = 5) -> list[list[float]]:
    """
    Remove crackle artifacts from stereo audio
    
    Args:
        audio: Stereo audio data as [[left_channel], [right_channel]]
        threshold_factor: Multiplier for standard deviation to detect spikes
        window_size: Size of median filter window (odd number)
    
    Returns:
        Filtered stereo audio
    """
    # Convert to numpy array for processing
    audio_array = np.array(audio)
    
    # Process each channel
    for channel in range(2):
        # Calculate local statistics
        rolling_std = np.std(audio_array[channel])
        threshold = rolling_std * threshold_factor
        
        # Detect crackles (large deviations)
        crackle_mask = abs(audio_array[channel]) > threshold

        crackle_count = sum(crackle_mask)
        print(f"Channel {channel}: {crackle_count} crackles detected")
        
        # Apply median filter to smooth out impulses
        filtered = signal.medfilt(audio_array[channel], window_size)
        
        # Replace only the detected crackle positions
        audio_array[channel][crackle_mask] = filtered[crackle_mask]
    
    return audio_array.tolist()

def decrackle_channel_2(channel_data: list[float], channel_name: str) -> list[float]:
    # Calculate amplitude envelope
    window = 5
    envelope = []
    for i in range(len(channel_data)):
        start = max(0, i - window)
        end = min(len(channel_data), i + window + 1)
        envelope.append(max(abs(x) for x in channel_data[start:end]))
    
    # Find sudden peaks in envelope (potential crackles)
    peaks, _ = find_peaks(envelope, 
                         height=0.1,          # Minimum height
                         distance=10,         # Minimum samples between peaks
                         prominence=0.05)     # Minimum prominence
    
    print(f"{channel_name} channel: {len(peaks)} crackles detected")
    
    # Create output array
    output = channel_data.copy()
    
    # Remove crackles by interpolation
    for peak in peaks:
        # Define window around crackle
        start = max(0, peak - 2)
        end = min(len(output), peak + 3)
        
        # Get clean samples before and after crackle
        x = [start-1, end+1]
        y = [output[start-1], output[end+1]]
        
        # Interpolate over crackle
        interp = interp1d(x, y, kind='linear')
        for i in range(start, end):
            output[i] = interp(i)
    
    return output

def stereo_decrackle_filter_2(stereo_audio: list[list[float]], threshold_factor: float = 2.0, window_size: int = 5) -> list[list[float]]:
    left_channel, right_channel = stereo_to_mono(stereo_audio)
    
    filtered_left = decrackle_channel_2(left_channel, "Left")
    filtered_right = decrackle_channel_2(right_channel, "Right")
    
    return mono_to_stereo(filtered_left, filtered_right)

def stereo_decrackle_filter(stereo_audio: list[list[float]], threshold_factor: float = 2.0, window_size: int = 5) -> list[list[float]]:
    def calculate_std(values: list[float]) -> float:
        mean = sum(values) / len(values)
        squared_diff_sum = sum((x - mean) ** 2 for x in values)
        return sqrt(squared_diff_sum / len(values))
    
    def decrackle_channel(channel_data: list[float]) -> list[float]:
        std = calculate_std(channel_data)
        min_threshold = 0.001
        threshold = max(std * threshold_factor, min_threshold)
        
        print(f"Channel statistics:")
        print(f"Standard deviation: {std}")
        print(f"Threshold: {threshold}")
        print(f"Max amplitude: {max(abs(x) for x in channel_data)}")
        
        crackle_mask = [abs(x) > threshold for x in channel_data]
        crackle_count = sum(1 for x in crackle_mask if x)
        print(f"{crackle_count} crackles detected\n")
        
        filtered = signal.medfilt(channel_data, window_size)
        filtered_channel = [
            filtered[i] if crackle_mask[i] else channel_data[i]
            for i in range(len(channel_data))
        ]
        
        return filtered_channel

    left_channel, right_channel = stereo_to_mono(stereo_audio)
    
    filtered_left = decrackle_channel(left_channel)
    filtered_right = decrackle_channel(right_channel)

    return mono_to_stereo(filtered_left, filtered_right)