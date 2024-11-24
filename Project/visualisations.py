from utils import *
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt

def show_stereo_spectrogram(
    stereo_audio: list[list[float]],
    fs: int,
    start_time: float = 0.0,
    end_time: float = None,
    log_scale: bool = True,
):
    start_index = int(start_time * fs)
    end_index = int(end_time * fs) if end_time is not None else len(stereo_audio)
    stereo_audio = stereo_audio[start_index:end_index]

    left_channel, right_channel = stereo_to_mono(stereo_audio)

    f_left, t_left, Sxx_left = signal.spectrogram(np.array(left_channel), fs)
    f_right, t_right, Sxx_right = signal.spectrogram(np.array(right_channel), fs)

    # Plot spectrograms
    fig = plt.figure(figsize=(12, 8))

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.pcolormesh(t_left, f_left, 10 * np.log10(Sxx_left), shading="gouraud")
    ax1.set_title("Left Channel Spectrogram")
    ax1.set_ylabel("Frequency [Hz]")
    ax1.set_xlabel("Time [sec]")
    if log_scale:
        ax1.set_yscale('log')

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.pcolormesh(t_right, f_right, 10 * np.log10(Sxx_right), shading="gouraud")
    ax2.set_title("Right Channel Spectrogram")
    ax2.set_ylabel("Frequency [Hz]")
    ax2.set_xlabel("Time [sec]")
    if log_scale:
        ax2.set_yscale('log')

    plt.show()
    return fig


def compare_stereo_spectrograms(
    stereo_audios: list[list[list[float]]],
    fs: int,
    title="Title",
    start_time=0.0,
    end_time=None,
    log_scale: bool = True,
):
    fig, axes = plt.subplots(len(stereo_audios), 2, figsize=(15, 6))
    fig.canvas.manager.set_window_title(title)

    for i, (row) in enumerate(axes):
        start_index = int(start_time * fs)
        end_index = (
            int(end_time * fs) if end_time is not None else len(stereo_audios[i])
        )
        stereo_audio = stereo_audios[i][start_index:end_index]

        left_channel, right_channel = stereo_to_mono(stereo_audio)

        f_left, t_left, Sxx_left = signal.spectrogram(np.array(left_channel), fs=fs)
        f_right, t_right, Sxx_right = signal.spectrogram(np.array(right_channel), fs=fs)

        row[0].pcolormesh(t_left, f_left, 10 * np.log10(Sxx_left), shading="gouraud")
        row[0].set_title(f"Left Channel Spectrogram {i}")
        row[0].set_ylabel("Frequency [Hz]")
        row[0].set_xlabel("Time [sec]")
        if log_scale:
            row[0].set_yscale('log')

        row[1].pcolormesh(t_right, f_right, 10 * np.log10(Sxx_right), shading="gouraud")
        row[1].set_title(f"Right Channel Spectrogram {i}")
        row[1].set_ylabel("Frequency [Hz]")
        row[1].set_xlabel("Time [sec]")
        if log_scale:
            row[1].set_yscale('log')

    plt.tight_layout()
    plt.show()
