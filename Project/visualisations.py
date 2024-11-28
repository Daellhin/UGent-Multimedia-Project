import matplotlib.ticker as mticker
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from utils import *


def show_stereo_spectrograms(
    stereo_audios: list[list[list[float]]],
    fs: int,
    title="Title",
    start_time=0.0,
    end_time=None,
    log_scale=False,
    colour_map="inferno",
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

        f_left, t_left, Sxx_left = signal.spectrogram(np.array(left_channel), fs)
        f_right, t_right, Sxx_right = signal.spectrogram(np.array(right_channel), fs)

        row[0].pcolormesh(t_left, f_left, 10 * np.log10(Sxx_left), cmap=colour_map)
        row[0].set_title(f"Left Channel Spectrogram {i}")
        row[0].set_ylabel("Frequency [Hz]")
        row[0].set_xlabel("Time [sec]")
        if log_scale:
            row[0].set_yscale("log")
            row[0].set_ylim([20, fs / 2])
            row[0].yaxis.set_major_formatter(mticker.ScalarFormatter())

        row[1].pcolormesh(t_right, f_right, 10 * np.log10(Sxx_right), cmap=colour_map)
        row[1].set_title(f"Right Channel Spectrogram {i}")
        row[1].set_ylabel("Frequency [Hz]")
        row[1].set_xlabel("Time [sec]")
        if log_scale:
            row[1].set_yscale("log")
            row[1].set_ylim([20, fs / 2])
            row[1].yaxis.set_major_formatter(mticker.ScalarFormatter())

    plt.tight_layout()
    plt.show()
