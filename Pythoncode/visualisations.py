import math

import cv2
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

    # Convert axes to 2D array if single row
    if len(stereo_audios) == 1:
        axes = np.array([axes])

    for i, (row) in enumerate(axes):
        start_index = int(start_time * fs)
        end_index = (
            int(end_time * fs) if end_time is not None else len(stereo_audios[i])
        )
        stereo_audio = stereo_audios[i][start_index:end_index]
        left_channel, right_channel = stereo_to_mono(stereo_audio)

        f_left, t_left, Sxx_left = signal.spectrogram(np.array(left_channel), fs)
        f_right, t_right, Sxx_right = signal.spectrogram(np.array(right_channel), fs)

        row[0].pcolormesh(t_left, f_left, 10 * np.log10(Sxx_left))
        row[0].set_title(f"Left Channel Spectrogram {i}")
        row[0].set_ylabel("Frequency [Hz]")
        row[0].set_xlabel("Time [sec]")
        if log_scale:
            row[0].set_yscale("log")
            row[0].set_ylim([20, fs / 2])
            row[0].yaxis.set_major_formatter(mticker.ScalarFormatter())

        row[1].pcolormesh(t_right, f_right, 10 * np.log10(Sxx_right))
        row[1].set_title(f"Right Channel Spectrogram {i}")
        row[1].set_ylabel("Frequency [Hz]")
        row[1].set_xlabel("Time [sec]")
        if log_scale:
            row[1].set_yscale("log")
            row[1].set_ylim([20, fs / 2])
            row[1].yaxis.set_major_formatter(mticker.ScalarFormatter())

    plt.tight_layout()
    plt.show()


def cumulative_histogram(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    cumhist = np.cumsum(hist).astype(np.float64)
    cumhist = cumhist / cumhist[-1]
    return cumhist


def show_histogram(image, image2, nameImg="", nameImg2=""):
    fig = plt.figure(figsize=(20, 15))
    ax = fig.add_subplot(321)
    ax.set_title(nameImg)
    ax0 = fig.add_subplot(322)
    ax0.set_title(nameImg2)
    ax1 = fig.add_subplot(323)
    ax1.set_title("Histogram of " + nameImg)
    ax2 = fig.add_subplot(324, sharex=ax1)
    ax2.set_title("Histogram of " + nameImg2)
    ax3 = fig.add_subplot(325, sharex=ax1)
    ax3.set_title("Cumulative Histogram of " + nameImg)
    ax4 = fig.add_subplot(326, sharex=ax1)
    ax4.set_title("Cumulative Histogram of " + nameImg2)
    ax1.set_xlim([0, 255])
    ax.imshow(image, cmap=plt.get_cmap("gray"))
    ax0.imshow(image2, cmap=plt.get_cmap("gray"))
    ax1.plot(cv2.calcHist([image], [0], None, [256], [0, 256]))
    ax2.plot(cv2.calcHist([image2], [0], None, [256], [0, 256]))
    ax3.plot(cumulative_histogram(image))
    ax4.plot(cumulative_histogram(image2))
    plt.show()


def magnitude_spectrum(fshift):
    magnitude_spectrum = np.log(np.abs(fshift) + 1)
    magnitude_spectrum -= np.min(magnitude_spectrum)
    magnitude_spectrum *= 255.0 / np.max(magnitude_spectrum)
    return magnitude_spectrum


def show_spectrum(image, original_image, figure_name):
    fft_image = np.fft.fft2(image)
    fft_image = np.fft.fftshift(fft_image)
    fft_original = np.fft.fft2(original_image)
    fft_original = np.fft.fftshift(fft_original)
    fig = plt.figure()
    fig.suptitle(figure_name, fontsize=14)
    ax1 = fig.add_subplot(221)
    ax1.set_title(figure_name + " before")
    ax2 = fig.add_subplot(222)
    ax2.set_title("FFT")
    ax3 = fig.add_subplot(223)
    ax3.set_title("FFT original")
    ax4 = fig.add_subplot(224)
    ax4.set_title(figure_name + " original")
    ax1.imshow(image, cmap="gray")
    ax2.imshow(magnitude_spectrum(fft_image), cmap="gray")
    ax3.imshow(magnitude_spectrum(fft_original), cmap="gray")
    ax4.imshow(original_image, cmap="gray")
    ax1.axis("off")
    ax2.axis("off")
    ax3.axis("off")
    ax4.axis("off")
    plt.tight_layout()
    plt.show()


def plot_lowpass_filter(
    b: list[float], a: list[float], fs: int, cutoff: int, order: int
):
    # Generate frequency response
    w, h = signal.freqz(b, a)
    freq = w * fs / (2 * math.pi)

    # Plot frequency response
    plt.figure(figsize=(10, 4))
    plt.plot(freq, 20 * np.log10(abs(h)))
    plt.grid(True)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude [dB]")
    plt.title(f"Lowpass Filter Frequency Response\nCutoff: {cutoff}Hz, Order: {order}")
    plt.axvline(cutoff, color="red", alpha=0.5)
    plt.axhline(-3, color="green", alpha=0.5)
    plt.show()


def plot_notch_filter(
    b: list[float],
    a: list[float],
    fs: int,
    frequency: int,
    quality: int,
    order: int,
    xlim=10000,
):
    # Generate frequency response
    freq, h = signal.freqz(b, a, fs=fs)

    # Plot frequency response
    plt.figure(figsize=(10, 4))
    plt.plot(freq, 20 * np.log10(abs(h)))
    plt.grid(True)
    plt.xlim([0, xlim])
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude [dB]")
    plt.title(
        f"Notch Filter Frequency Response\nFrequency: {frequency}Hz, Quality: {quality}, Order: {order}"
    )
    plt.axvline(frequency, color="red", alpha=0.5)
    plt.show()
