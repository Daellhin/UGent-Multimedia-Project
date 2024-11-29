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


def stereo_wiener_filter(audio_samples: list[list[float]], mysize=15):
    left_channel, right_channel = stereo_to_mono(audio_samples)

    filtered_left = signal.wiener(left_channel, mysize=mysize)
    filtered_right = signal.wiener(right_channel, mysize=mysize)

    filtered_audio = mono_to_stereo(filtered_left, filtered_right)
    return filtered_audio


def create_audio_clip(samples: list[list[float]], fps: int):
    def make_frame(t: float):
        return np.array(samples)

    return AudioClip(make_frame, duration=len(samples) / fps)


def stereo_reduce_noise_filter(
    audio_samples: list[list[float]],
    fs: int,
    stationary=False,
    n_fft=1024,
):
    """
    Reduce noise using spectral gating with noise reduce
    """
    left_channel, right_channel = stereo_to_mono(audio_samples)
    filtered_left = nr.reduce_noise(
        left_channel, fs, stationary=stationary, n_fft=n_fft, use_tqdm=True
    )
    filtered_right = nr.reduce_noise(
        right_channel, fs, stationary=stationary, n_fft=n_fft, use_tqdm=True
    )
    return mono_to_stereo(filtered_left, filtered_right)


def apply_audio_filters(
    audio_samples: list[list[float]], fs: int, filter_size: int = 5
) -> list[list[float]]:
    filtered_audio_1 = sterio_notch_filter(audio_samples, fs, 100, 1, 2)
    filtered_audio_2 = stereo_butterworth_filter(
        "lowpass", filtered_audio_1, fs, 5500, 5
    )
    filtered_audio_3 = stereo_reduce_noise_filter(filtered_audio_2, fs)

    show_stereo_spectrograms(
        [audio_samples, filtered_audio_1, filtered_audio_2, filtered_audio_3],
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
    processed_audio = apply_audio_filters(audio_samples, fs)

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
