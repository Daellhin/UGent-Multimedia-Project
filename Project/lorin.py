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
from scipy.io import wavfile
from utils import *
from visualisations import *

show_images = True


def sterio_notch_filter(
    stereo_audio: list[list[float]], fs: int, frequency: float, Q: float = 30.0, order=1
):
    """
    Q: Quality factor (higher Q = narrower notch)
    order: Filter order (higher = sharper cutoff)
    """
    left_channel, right_channel = stereo_to_mono(stereo_audio)

    b, a = signal.iirnotch(frequency, Q, fs)
    for _ in range(order):
        left_channel = signal.filtfilt(b, a, left_channel)
        right_channel = signal.filtfilt(b, a, right_channel)

    return mono_to_stereo(left_channel, right_channel)


def stereo_butterworth_filter(
    type: Literal["lowpass", "highpass"],
    stereo_audio: list[list[float]],
    fs: int,
    cutoff: float,
    order=5,
):
    left_channel, right_channel = stereo_to_mono(stereo_audio)

    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype=type)

    filtered_left = signal.lfilter(b, a, left_channel)
    filtered_right = signal.lfilter(b, a, right_channel)

    return mono_to_stereo(filtered_left, filtered_right)


# def stereo_decrackle_filter(audio: list[list[float]], threshold_factor: float = 3.0, window_size: int = 5) -> list[list[float]]:
#     """
#     Remove crackle artifacts from stereo audio
    
#     Args:
#         audio: Stereo audio data as [[left_channel], [right_channel]]
#         threshold_factor: Multiplier for standard deviation to detect spikes
#         window_size: Size of median filter window (odd number)
    
#     Returns:
#         Filtered stereo audio
#     """
#     # Convert to numpy array for processing
#     audio_array = np.array(audio)
    
#     # Process each channel
#     for channel in range(2):
#         # Calculate local statistics
#         rolling_std = np.std(audio_array[channel])
#         threshold = rolling_std * threshold_factor
        
#         # Detect crackles (large deviations)
#         crackle_mask = abs(audio_array[channel]) > threshold

#         crackle_count = sum(crackle_mask)
#         print(f"Channel {channel}: {crackle_count} crackles detected")
        
#         # Apply median filter to smooth out impulses
#         filtered = signal.medfilt(audio_array[channel], window_size)
        
#         # Replace only the detected crackle positions
#         audio_array[channel][crackle_mask] = filtered[crackle_mask]
    
#     return audio_array.tolist()
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
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

def mono_spectral_noise_filter(
    audio: list[float],
    fs: int,
    noise_estimation_time=0.01,
    kernel_size=(1, 15),
):
    """
    Estimates the noise power in each frequency using the first samples of the audio.
    If noise power is greater than the magnitude of the frequency, the frequency is set to 0.
    """
    _, _, stft = signal.stft(audio, fs)
    magnitude = abs(stft)
    phase = np.exp(1.0j * np.angle(stft))

    noise_estimation_samples = int(fs * noise_estimation_time)
    noise_power = np.mean(magnitude[:, :noise_estimation_samples], axis=1)
    noise_power = noise_power * 1.1
    binary_mask = (magnitude > noise_power[:, None]).astype(float)
    binary_mask = signal.medfilt(binary_mask, kernel_size)
    frequencies_filtered = magnitude * binary_mask

    stft_filtered = frequencies_filtered * phase
    _, audio_filtered = signal.istft(stft_filtered, fs)
    return audio_filtered


def stereo_spectral_noise_estimation_filter(
    stereo_audio: list[list[float]],
    fs: int,
    noise_estimation_time=0.01,
    kernel_size=(1, 15),
):
    left_channel, right_channel = stereo_to_mono(stereo_audio)

    filtered_left = mono_spectral_noise_filter(
        left_channel, fs, noise_estimation_time, kernel_size
    )
    filtered_right = mono_spectral_noise_filter(
        right_channel, fs, noise_estimation_time, kernel_size
    )

    return mono_to_stereo(filtered_left, filtered_right)


def stereo_wiener_filter(audio_samples: np.ndarray, mysize=15):
    # Split stereo audio into left and right channels
    left_channel, right_channel = stereo_to_mono(audio_samples)

    # Apply Wiener filter to each channel
    filtered_left = signal.wiener(left_channel, mysize=mysize)
    filtered_right = signal.wiener(right_channel, mysize=mysize)

    # Combine the channels back into stereo
    filtered_audio = np.vstack((filtered_left, filtered_right)).T
    return filtered_audio


def create_audio_clip(samples, fps):
    def make_frame(t):
        return np.array(samples)

    return AudioClip(make_frame, duration=len(samples) / fps)


def stereo_noise_reduce_filter(
    audio_samples: list[list[float]], fs: int, stationary=False, n_fft=1024,
):
    left_channel, right_channel = stereo_to_mono(audio_samples)
    filtered_left = nr.reduce_noise(
        left_channel, fs, stationary=stationary, n_fft=n_fft, use_tqdm=True
    )
    filtered_right = nr.reduce_noise(
        right_channel, fs, stationary=stationary, n_fft=n_fft, use_tqdm=True
    )
    return mono_to_stereo(filtered_left, filtered_right)

def stereo_noise_reduce_filter_clip(
    audio_samples: list[list[float]], fs: int, stationary=False, n_fft=1024,
):
    left_channel, right_channel = stereo_to_mono(audio_samples)
    filtered_left = nr.reduce_noise(
        left_channel, fs, stationary=stationary, n_fft=n_fft, use_tqdm=True, y_noise=left_channel[0:fs*0.25]
    )
    filtered_right = nr.reduce_noise(
        right_channel, fs, stationary=stationary, n_fft=n_fft, use_tqdm=True, y_noise=right_channel[0:fs*0.25]
    )
    return mono_to_stereo(filtered_left, filtered_right)


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


def reduce_noise(
    audio_samples: list[list[float]], fs: int, filter_size: int = 5
) -> list[list[float]]:
    filtered_audio_1 = sterio_notch_filter(audio_samples, fs, 100, 1, 2)
    filtered_audio_2 = stereo_butterworth_filter(
        "lowpass", filtered_audio_1, fs, 5500, 5
    )
    # filtered_audio_3 = stereo_wiener_filter(filtered_audio_2, 15)
    # filtered_audio_3 = stereo_spectral_noise_estimation_filter(
    #     filtered_audio_2, fs, 0.01
    # )
    # filtered_audio_3 = stereo_wavelet_denoise(filtered_audio_2)
    #filtered_audio_3 = stereo_decrackle_filter(filtered_audio_2)
    filtered_audio_3 = stereo_noise_reduce_filter(filtered_audio_2, fs)
    # filtered_audio_3 = stereo_median_filter(filtered_audio_2, 15)
    # filtered_audio_4 = stereo_wavelet_denoise(filtered_audio_3)
    # filtered_audio_4 = stereo_decrackle_filter_2(filtered_audio_3)
    
    show_stereo_spectrograms(
        [audio_samples,filtered_audio_3],
        fs,
        "Noise reduction Spectogram",
        end_time=5,
        log_scale=False,
    )

    return filtered_audio_3


def process_frame(
    get_frame: Callable[[float], np.ndarray], time: float, frame_duration: float
):
    """
    Process frame callback:
    - use get_frame with time to get current frame
    - use get_frame with multiple times to get multiple frames(much slower)
    """
    # -- Get frame(s) --
    frame = get_frame(time)
    # times = np.arange(0, 4) * frame_duration
    # frames = list(map(get_frame, times))

    # -- Modify frame --

    # -- Debug images --
    global show_images
    if show_images:
        im_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Frame", im_bgr)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        show_images = not show_images

    return frame


def stereo_calculate_MSE(
    original: list[list[float]], filtered: list[list[float]], n_samples=200000
):
    """
    Returns Mean Squared quadratic Error(MSE) for both chanels for the n first samples
    """
    left_original, right_original = stereo_to_mono(original)
    left_filtered, right_filtered = stereo_to_mono(filtered)
    mse_left = (
        sum((left_filtered[i] - left_original[i]) ** 2 for i in range(n_samples))
        / n_samples
    )
    mse_right = (
        sum((right_filtered[i] - right_original[i]) ** 2 for i in range(n_samples))
        / n_samples
    )
    return (mse_left, mse_right)


def process_video(input_path: str, output_path: str):
    print("Processing: ", input_path)

    video = VideoFileClip(input_path)
    # -- Video processing --
    print(f"Processing video (s={video.duration})")
    # frame_duration = 1 / video.fps  # frame duration in seconds
    # processed_video: VideoFileClip = video.fl(
    #     lambda gf, t: process_frame(gf, t, frame_duration)
    # )
    # cv2.destroyAllWindows()
    # processed_video = video  # for debuging audio without video

    # -- Audio processing --
    # Load audio
    fs = video.audio.fps
    print(f"Loading audio frames (fs={fs}) (s={video.audio.duration})")
    audio_samples: list[list[float]] = list(video.audio.to_soundarray())

    # Process audio
    print(f"Applying audio filters")
    processed_audio = reduce_noise(audio_samples, fs)

    # Audio output
    audio_path = f"{output_path}.wav"
    print(f"Writing audio to:", audio_path)
    wavfile.write(audio_path, fs, np.array(processed_audio, dtype=np.float32))

    # t = AudioFileClip("output/output.mp4.wav")
    # result: VideoFileClip = processed_video.set_audio(t)
    # #play_stereo_audio(processed_audio)

    # # -- Output results --
    # result.write_videofile(output_path)


def compair_audio(original_path: str, filtered_path: str):
    original = AudioFileClip(original_path)
    filtered = AudioFileClip(filtered_path)

    mse = stereo_calculate_MSE(original.to_soundarray(), filtered.to_soundarray())
    print(f"MSE| chanel 0: {mse[0]}, chanel 1: {mse[1]}")


def main():
    process_video(
        "DegradedVideos/archive_2017-01-07_President_Obama's_Weekly_Address.mp4",
        "output/output.mp4",
    )

    # compair_audio("output/original.mp4.wav", "output/input.mp4.wav")
    # compair_audio("output/original.mp4.wav", "output/output.mp4.wav")


if __name__ == "__main__":
    main()

# https://github.com/tennisonliu/noise_reduction/blob/master/spectral_gating.ipynb
# https://www.google.com/search?client=firefox-b-d&q=python+spectral+gating

# https://www.youtube.com/watch?v=M9SpsrIolxo
