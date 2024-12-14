import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import reduce
from math import *
from statistics import *
from typing import Literal

import cv2
import noisereduce as nr
import numpy as np
from moviepy import AudioFileClip, VideoFileClip
from scipy import signal
from scipy.io import wavfile
from skimage import metrics
from utils import *
from visualisations import *


@dataclass
class NotchFilter:
    frequency: int
    Q: float
    order: int


@dataclass
class ButterworthFilters:
    type: Literal["lowpass", "highpass"]
    cuttof: int
    order: int


@dataclass
class ReduceNoiseFilters:
    stationary: bool
    n_fft: int
    strength: float


def sterio_notch_filter(
    stereo_audio: list[list[float]],
    fs: int,
    frequency: float,
    Q: float = 30.0,
    order=1,
    debug=False,
):
    """
    Removes a specific frequency using IIR notch filter
    Q: Quality factor (higher Q = narrower notch)
    order: Filter order (higher = sharper cutoff)
    """
    left_channel, right_channel = stereo_to_mono(stereo_audio)

    b_cascade, a_cascade = signal.iirnotch(frequency, Q, fs)
    b = reduce(np.convolve, [b_cascade] * order)
    a = reduce(np.convolve, [a_cascade] * order)

    if debug:
        plot_notch_filter(b, a, fs, frequency, Q, order)

    left_channel = signal.filtfilt(b, a, left_channel)
    right_channel = signal.filtfilt(b, a, right_channel)

    return mono_to_stereo(left_channel, right_channel)


def stereo_butterworth_filter(
    stereo_audio: list[list[float]],
    fs: int,
    type: Literal["lowpass", "highpass"],
    cutoff: float,
    order=5,
    debug=False,
):
    """
    Removes high or low frequencies using butterworth filter
    """
    left_channel, right_channel = stereo_to_mono(stereo_audio)

    nyquist = 0.5 * fs
    normalised_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normalised_cutoff, btype=type)

    if debug:
        plot_lowpass_filter(b, a, fs, cutoff, order)

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
    """
    Reduce noise using spectral gating, own implementation. Less succesful than noisereduce
    """
    left_channel, right_channel = stereo_to_mono(stereo_audio)

    filtered_left = mono_spectral_noise_filter(
        left_channel, fs, noise_estimation_time, kernel_size
    )
    filtered_right = mono_spectral_noise_filter(
        right_channel, fs, noise_estimation_time, kernel_size
    )

    return mono_to_stereo(filtered_left, filtered_right)


def stereo_wiener_filter(audio_samples: list[list[float]], mysize=15):
    """
    Reduce noise using wiener filter. Not succesful!
    """
    left_channel, right_channel = stereo_to_mono(audio_samples)

    filtered_left = signal.wiener(left_channel, mysize=mysize)
    filtered_right = signal.wiener(right_channel, mysize=mysize)

    filtered_audio = mono_to_stereo(filtered_left, filtered_right)
    return filtered_audio


def stereo_reduce_noise_filter(
    audio_samples: list[list[float]],
    fs: int,
    stationary=False,
    n_fft=1024,
    use_tqdm=False,
    strength=1.0,
):
    """
    Reduce noise using spectral gating with noise reduce
    """
    left_channel, right_channel = stereo_to_mono(audio_samples)

    filtered_left = nr.reduce_noise(
        left_channel,
        fs,
        stationary,
        prop_decrease=strength,
        n_fft=n_fft,
        use_tqdm=use_tqdm,
    )
    filtered_right = nr.reduce_noise(
        right_channel,
        fs,
        stationary,
        prop_decrease=strength,
        n_fft=n_fft,
        use_tqdm=use_tqdm,
    )

    return mono_to_stereo(filtered_left, filtered_right)


def amplify_audio(audio_channel: list[float], amplification_factor: float):
    return [sample * amplification_factor for sample in audio_channel]


def sterio_amplifie_audio(
    audio_samples: list[list[float]], amplification_factor: float
):
    left_channel, right_channel = stereo_to_mono(audio_samples)

    adjusted_data_left = amplify_audio(left_channel, amplification_factor)
    adjusted_data_right = amplify_audio(right_channel, amplification_factor)

    return mono_to_stereo(adjusted_data_left, adjusted_data_right)


def apply_audio_filters(
    audio_samples: list[list[float]],
    fs: int,
    notch_filters: list[NotchFilter] = [],
    butterworth_filters: list[ButterworthFilters] = [],
    reduce_noise_filters: list[ReduceNoiseFilters] = [],
    amplification_factor=1.0,
    debug=False,
) -> list[list[float]]:
    filtered_audio_1 = reduce(
        lambda x, f: sterio_notch_filter(x, fs, f.frequency, f.Q, f.order, debug),
        notch_filters,
        audio_samples,
    )
    filtered_audio_2 = reduce(
        lambda x, f: stereo_butterworth_filter(x, fs, f.type, f.cuttof, f.order, debug),
        butterworth_filters,
        filtered_audio_1,
    )
    filtered_audio_3 = reduce(
        lambda x, f: stereo_reduce_noise_filter(
            x, fs, f.stationary, f.n_fft, True, f.strength
        ),
        reduce_noise_filters,
        filtered_audio_2,
    )
    filtered_audio_4 = sterio_amplifie_audio(filtered_audio_3, amplification_factor)

    if debug:
        show_stereo_spectrograms(
            [filtered_audio_3],
            fs,
            "Noise reduction Spectogram",
            end_time=5,
            log_scale=False,
        )

    return filtered_audio_4


def compare_audio(
    audio_samples_orig: list[list[float]],
    audio_samples_degr: list[list[float]],
    audio_samples_proc: list[list[float]],
):
    """
    Berekend:
    - Mean Squared Error (MSE): [0, ∞] -> 0 is perfect
    - Peak Signal to Noise Ratio (PSNR): [0, ∞] -> ∞ is perfect
    """
    length_proc = len(audio_samples_proc)
    length_orig = len(audio_samples_orig)

    if length_proc != length_orig:
        min_length = min(length_proc, length_orig)
        print(
            f"Warning: Audio lengths do not match, processed: {length_proc}, original: {length_orig}. Using minimum: {min_length}"
        )
        audio_samples_orig = audio_samples_orig[:min_length]
        audio_samples_degr = audio_samples_degr[:min_length]
        audio_samples_proc = audio_samples_proc[:min_length]

    left_channel_proc = np.array(stereo_to_mono(audio_samples_proc)[0])
    left_channel_degr = np.array(stereo_to_mono(audio_samples_degr)[0])
    left_channel_orig = np.array(stereo_to_mono(audio_samples_orig)[0])

    mse_degr = metrics.mean_squared_error(left_channel_orig, left_channel_degr)
    mse_proc = metrics.mean_squared_error(left_channel_orig, left_channel_proc)
    psnr_degr = metrics.peak_signal_noise_ratio(left_channel_orig, left_channel_degr)
    psnr_proc = metrics.peak_signal_noise_ratio(left_channel_orig, left_channel_proc)

    print(f"MSE: {mse_degr} -> {mse_proc}")
    print(f"PSNR: {psnr_degr} -> {psnr_proc}")
    return (mse_degr, mse_proc), (psnr_degr, psnr_proc)


def process_audio(
    input_path: str,
    output_path="output/output.mp4",
    input_path_original: str = None,
    notch_filters: list[NotchFilter] = [],
    butterworth_filters: list[ButterworthFilters] = [],
    reduce_noise_filters: list = [],
    amplification_factor=1.0,
    debug=False,
):
    print("-- Processing audio of: ", input_path)

    # Load audio
    audio = AudioFileClip(input_path)
    fs = audio.fps
    print(f"Loading audio frames (fs={fs}) (s={audio.duration})")
    audio_samples: list[list[float]] = list(audio.to_soundarray())

    # Process audio
    print(f"Applying audio filters: ")
    processed_audio = apply_audio_filters(
        audio_samples,
        fs,
        notch_filters,
        butterworth_filters,
        reduce_noise_filters,
        amplification_factor,
        debug,
    )

    # Audio output
    audio_path = f"{output_path}.wav"
    print(f"Writing audio to:", audio_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    wavfile.write(audio_path, fs, np.array(processed_audio, dtype=np.float32))

    if input_path_original:
        audio_original = AudioFileClip(input_path_original)
        fs_original = audio_original.fps
        print(
            f"Loading original audio frames (fs={fs_original}) (s={audio_original.duration})"
        )

        audio_samples_original: list[list[float]] = list(audio_original.to_soundarray())
        compare_audio(audio_samples_original, audio_samples, processed_audio)

    return AudioFileClip(audio_path)


def combine_audio_with_video(
    audio: AudioFileClip, video: VideoFileClip, output_path: str
):
    print("-- Combining audio and video: ", audio.filename, video.filename)
    # -- Combine video and audio --
    result: VideoFileClip = video.with_audio(audio)
    result.write_videofile(output_path)


def main():
    start_time = time.time()
    debug = False
    # -- Degraded videos --
    processed_audio = process_audio(
        "DegradedVideos/archive_2017-01-07_President_Obama's_Weekly_Address.mp4",
        "output/output_obama.mp4",
        "SourceVideos/2017-01-07_President_Obama's_Weekly_Address.mp4",
        notch_filters=[NotchFilter(100, 30, 2)],
        butterworth_filters=[ButterworthFilters("lowpass", 5500, 5)],
        reduce_noise_filters=[ReduceNoiseFilters(False, 2048, 1)],
        amplification_factor=2.0,
        debug=debug,
    )
    # process_video(
    #     "DegradedVideos/archive_20240709_female_common_yellowthroat_with_caterpillar_canoe_meadows.mp4",
    #     "output/output_yellowthroat.mp4",
    #     notch_filters=[NotchFilter(100, 1, 2)],
    #     reduce_noise_filters=[ReduceNoiseFilters(False, 2048*4, 1)],
    #     amplification_factor=1.5,
    #     debug=debug,
    # )
    # process_video(
    #     "DegradedVideos/archive_Henry_Purcell__Music_For_a_While__-_Les_Arts_Florissants,_William_Christie.mp4",
    #     "output/output_arts_florissants.mp4",
    #     notch_filters=[NotchFilter(100, 1, 2)],
    #     butterworth_filters=[ButterworthFilters("lowpass", 10000, 5)],
    #     reduce_noise_filters=[ReduceNoiseFilters(True, 2048*4, 1)],
    #     amplification_factor=2.0,
    #     debug=debug,
    # )
    # process_video(
    #     "DegradedVideos/archive_Jasmine_Rae_-_Heartbeat_(Official_Music_Video).mp4",
    #     "output/output_heartbeat.mp4",
    #     notch_filters=[NotchFilter(100, 1, 2)],
    #     butterworth_filters=[ButterworthFilters("lowpass", 5500, 5)],
    #     reduce_noise_filters=[ReduceNoiseFilters(False, 2048*4, 1)],
    #     amplification_factor=1.0,
    #     debug=debug,
    # )
    # process_video(
    #     "DegradedVideos/archive_Robin_Singing_video.mp4",
    #     "output/output_robin.mp4",
    #     notch_filters=[NotchFilter(100, 1, 2)],
    #     butterworth_filters=[ButterworthFilters("lowpass", 5500, 5)],
    #     reduce_noise_filters=[ReduceNoiseFilters(False, 2048*4, 1)],
    #     amplification_factor=1.0,
    #     debug=debug,
    # )

    # # -- Archive videos --
    # process_video(
    #     "ArchiveVideos/Apollo_11_Landing_-_first_steps_on_the_moon.mp4",
    #     "output/output_apollo.mp4",
    #     notch_filters=[NotchFilter(190, 1, 2), NotchFilter(110, 1, 2), NotchFilter(50, 1, 2)],
    #     reduce_noise_filters=[ReduceNoiseFilters(False, 2048*2, 1)],
    #     amplification_factor=1.0,
    #     debug=debug,
    # )
    # process_video(
    #     "ArchiveVideos/Breakfast-at-tiffany-s-official®-trailer-hd.mp4",
    #     "output/output_tiffany.mp4",
    #     debug=debug,
    # )
    # process_video(
    #     "ArchiveVideos/Edison_speech,_1920s.mp4",
    #     "output/output_edison.mp4",
    #     notch_filters=[NotchFilter(100, 1, 1)],
    #     butterworth_filters=[ButterworthFilters("lowpass", 5000, 7)],
    #     reduce_noise_filters=[ReduceNoiseFilters(False, 2048, 1)],
    #     amplification_factor=1.0,
    #     debug=debug,
    # )
    # process_video(
    #     "ArchiveVideos/President_Kennedy_speech_on_the_space_effort_at_Rice_University,_September_12,_1962.mp4",
    #     "output/output_kennedy.mp4",
    #     notch_filters=[NotchFilter(50, 1, 1)],
    #     debug=debug,
    # )
    # process_video(
    #     "ArchiveVideos/The_Dream_of_Kings.mp4",
    #     "output/output_king.mp4",
    #     butterworth_filters=[ButterworthFilters("lowpass", 5500, 7)],
    #     reduce_noise_filters=[ReduceNoiseFilters(False, 2048, 1)],
    #     amplification_factor=2.0,
    #     debug=debug,
    # )

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Audio processing completed in {execution_time:.2f} seconds")


if __name__ == "__main__":
    main()
