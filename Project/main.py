import functools
import time
from dataclasses import dataclass
from gc import enable

import cv2
import numpy as np
from cv2.typing import MatLike

from lorin import *
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy.optimize import minimize
from skimage import metrics
from tqdm import tqdm
from visualisations import *


@dataclass
class ColorParams:
    # eerste filter voor bewerking
    FilterSize: int = 3
    # kleuraanpassing met gausiaanse vorm (hoog aan randen)
    GaussianSize: int = 1080
    UGaussianAdjust: float = 0
    VGaussianAdjust: float = 0
    YGaussianAdjust: float = 0
    # YUV-kleurcorrecties
    UMultiply: float = 1
    VMultiply: float = 1
    YSubstract: int = 0
    USubstract: int = 0
    VSubstract: int = 0
    # HSV-kleurcorrecties
    SaturationAdd: int = 0
    SaturationMultiply: float = 1
    ValueAdd: int = 0
    ValueMultiply: float = 1


@dataclass
class Enablers:
    kleurrek: bool = False
    show_color_steps: bool = False
    show_processed_frame: bool = False
    evaluate: bool = False
    stabilize: bool = False
    stabilize_colors: bool = False
    stabilize_only_x: bool = False
    stabilize_limit: bool = False
    debug_audio: bool = False
    remove_vertical_lines: bool = False
    bijsnijden: bool = False


# Pre defined dataclasses
noEffectColor = ColorParams()
allOff = Enablers()


@functools.lru_cache(maxsize=32)
def create_color_mask(shape, GaussianSize):
    rows, cols = shape
    kernel_x = cv2.getGaussianKernel(cols, GaussianSize)
    kernel_y = cv2.getGaussianKernel(rows, GaussianSize)
    kernel = kernel_y * kernel_x.T
    mask = 1 - kernel / np.linalg.norm(kernel)
    return cv2.normalize(mask, None, 0, 1, cv2.NORM_MINMAX)


def color_adjust(
    frame: MatLike,
    frameOrig: MatLike,
    params: ColorParams,
    show_steps=False,
):
    """
    Execution time: 0.1s - 0.2s
    """
    frame = cv2.blur(frame, (params.FilterSize, params.FilterSize))
    # YUV modifier - kringverzwakking
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv)

    if show_steps:
        cv2.imwrite("output/start_Frame.jpg", frame)
        yuv_or = cv2.cvtColor(frameOrig, cv2.COLOR_BGR2YUV)
        yo, uo, vo = cv2.split(yuv_or)
        show_histogram(y, yo, "Y", "Y Original")
        show_histogram(u, uo, "U", "U original")
        show_histogram(v, vo, "V", "V Original")

    # y = scipy.ndimage.median_filter(y, (3,3))

    mask = create_color_mask(v.shape, params.GaussianSize)
    if show_steps:
        plt.imshow(mask)
        plt.show()
    y = cv2.multiply(
        y.astype(np.float64),
        1 - params.YGaussianAdjust / 2 + params.YGaussianAdjust * mask,
    )
    u = cv2.multiply(u.astype(np.float64), 1 + params.UGaussianAdjust * mask)
    v = cv2.multiply(v.astype(np.float64), 1 + params.VGaussianAdjust * mask)
    y = cv2.subtract(y, params.YSubstract)
    u = cv2.multiply(u, params.UMultiply)
    u = cv2.subtract(u, params.USubstract)
    v = cv2.multiply(v, params.VMultiply)
    v = cv2.subtract(v, params.VSubstract)
    y = np.clip(y, 0, 255).astype(np.uint8)
    u = np.clip(u, 0, 255).astype(np.uint8)
    v = np.clip(v, 0, 255).astype(np.uint8)
    if show_steps:
        yuv_or = cv2.cvtColor(frameOrig, cv2.COLOR_BGR2YUV)
        yo, uo, vo = cv2.split(yuv_or)
        show_histogram(y, yo, "Y edit", "Y Original")
        show_histogram(u, uo, "U edit", "U original")
        show_histogram(v, vo, "V edit", "V Original")

    frame = cv2.merge((y, u, v))
    frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR)
    # BGR analyser
    if show_steps:
        cv2.imwrite("output/YUV-edit_Frame.jpg", frame)
        b, g, r = cv2.split(frame)
        bo, go, ro = cv2.split(frameOrig)
        show_histogram(r, ro, "Red", "Red Original")
        show_spectrum(r, ro, "Red")
        show_histogram(g, go, "Green", "Green Original")
        show_spectrum(g, go, "Green")
        show_histogram(b, bo, "Blue", "Blue Original")
        show_spectrum(b, bo, "Blue")

    # HSV modifiers
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    if show_steps:
        cv2.imwrite("output/BGR-edit_frame.jpg", frame)
        hsvOrig = cv2.cvtColor(frameOrig, cv2.COLOR_BGR2HSV)
        ho, so, vo = cv2.split(hsvOrig)
        show_histogram(h, ho, "Hue", "Hue Original")
        show_histogram(v, vo, "Value", "Value Original")
        show_histogram(s, so, "Saturation", "Saturation Original")

    v = cv2.multiply(v, params.ValueMultiply)
    v = cv2.add(v, params.ValueAdd)
    s = cv2.multiply(s, params.SaturationMultiply)
    s = cv2.add(s, params.SaturationAdd)

    h = np.clip(h, 0, 255).astype(np.uint8)
    v = np.clip(v, 0, 255).astype(np.uint8)
    s = np.clip(s, 0, 255).astype(np.uint8)
    final_hsv = cv2.merge((h, s, v))
    frame = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    if show_steps:
        cv2.imwrite("output/eind_frame.jpg", frame)
        hsvOrig = cv2.cvtColor(frameOrig, cv2.COLOR_BGR2HSV)
        ho, so, vo = cv2.split(hsvOrig)
        show_histogram(h, ho, "Hue", "Hue Original")
        show_histogram(v, vo, "Value", "Value Original")
        show_histogram(s, so, "Saturation", "Saturation Original")
    return frame


@functools.lru_cache(maxsize=32)
def radial_shift_map(height: int, width: int, scale_factor: float, power=1.0):
    """Maak een verschuivingskaart op basis van een radiale functie."""
    # Definieer het centrum van de afbeelding (assumeer centraal)
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    center_x, center_y = width // 2, height // 2
    dx = x - center_x
    dy = y - center_y
    r = (
        np.sqrt(dx**2 + dy**2) + 1e-8
    )  # Voeg kleine waarde toe om deling door 0 te vermijden
    # Bereken de verschuiving
    shift_x = scale_factor * (dx / r) * (r**power)
    shift_y = scale_factor * (dy / r) * (r**power)

    return shift_x, shift_y


def apply_radial_shift(channel: MatLike, shift_x: np.ndarray, shift_y: np.ndarray):
    """Pas de verschuivingen toe op een kleurkanaal."""
    map_x, map_y = np.meshgrid(np.arange(channel.shape[1]), np.arange(channel.shape[0]))
    map_x = (map_x - shift_x).astype(np.float32)
    map_y = (map_y - shift_y).astype(np.float32)
    corrected = cv2.remap(
        channel,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return corrected


def optimaliseer_kleurrek(frame: MatLike):
    """
    Average execution time: 0.2s
    """
    # Splits de afbeelding in BGR-kanalen
    b, g, r = cv2.split(frame)

    # Parameters voor verschuivingen
    scale_factor_red = 0.004  # Pas aan op basis van quiver-plot of experimenten
    scale_factor_blue = -0.003

    # Bereken verschuivingskaarten voor rode en blauwe kanalen
    height, width = g.shape
    shift_x_r, shift_y_r = radial_shift_map(height, width, scale_factor_red)
    shift_x_b, shift_y_b = radial_shift_map(height, width, scale_factor_blue)

    # Corrigeer de rode en blauwe kanalen
    aligned_r = apply_radial_shift(r, shift_x_r, shift_y_r)
    aligned_b = apply_radial_shift(b, shift_x_b, shift_y_b)

    # Combineer de gecorrigeerde kanalen
    aligned_image = cv2.merge((aligned_b, g, aligned_r))

    return aligned_image


def stabiliseer_frames(
    prev_frame: MatLike, curr_frame: MatLike, limit: bool, only_x: bool
):
    # Convert images to grayscale for feature detection
    gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and compute descriptors
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # Create brute-force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(des1, des2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Select good matches
    good_matches = matches[:40]

    # Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Estimate translation matrix
    try:
        translation_matrix, _ = cv2.estimateAffinePartial2D(
            dst_pts, src_pts, ransacReprojThreshold=10
        )
    except:
        print("Er liep iets mis")
        return curr_frame, None

    #############
    if limit:
        # Creëer een nieuwe rotatiematrix met vaste hoek zonder draaiing
        translation_matrix[:2, :2] = np.array([[1, 0], [0, 1]])

        # Beperk de verplaatsing in x en y richting
        if translation_matrix[0, 2] > 10:
            translation_matrix[0, 2] = 10
        elif translation_matrix[0, 2] < -10:
            translation_matrix[0, 2] = -10

    if only_x:
        # Forceer y-translatie naar 0, behoud x-translatie
        translation_matrix[1, 2] = 0

    #############

    # Apply translation to the color image
    # aligned_img2 = cv2.warpAffine(img2, translation_matrix, (img1.shape[1], img1.shape[0]))

    aligned_img2 = cv2.warpAffine(
        curr_frame,
        translation_matrix,
        (prev_frame.shape[1], prev_frame.shape[0]),
        borderMode=cv2.BORDER_REPLICATE,  # Herhaalt de randpixels gespiegeld
    )

    return aligned_img2, translation_matrix


def align_and_stabilize_frame(prev_frame: MatLike, curr_frame: MatLike):
    """
    Stabiliseert een frame en aligneert de kleuren tussen frames.
    """
    # Split de frames in kleurkanalen
    b1, g1, r1 = cv2.split(prev_frame)
    b2, g2, r2 = cv2.split(curr_frame)

    def calculate_alignment_error(params, target_channels, source_channels):
        """
        Berekent aligneringsfout tussen kleurkanalen.
        """
        b_dx, b_dy, g_dx, g_dy, r_dx, r_dy = params

        # Verschuif kleurkanalen
        b_aligned = cv2.warpAffine(
            source_channels[0],
            np.float32([[1, 0, b_dx], [0, 1, b_dy]]),
            (target_channels[0].shape[1], target_channels[0].shape[0]),
        )
        g_aligned = cv2.warpAffine(
            source_channels[1],
            np.float32([[1, 0, g_dx], [0, 1, g_dy]]),
            (target_channels[1].shape[1], target_channels[1].shape[0]),
        )
        r_aligned = cv2.warpAffine(
            source_channels[2],
            np.float32([[1, 0, r_dx], [0, 1, r_dy]]),
            (target_channels[2].shape[1], target_channels[2].shape[0]),
        )

        # Bereken verschil tussen gealigneerde en doel-kanalen
        diff = (
            np.abs(target_channels[0].astype(float) - b_aligned.astype(float))
            + np.abs(target_channels[1].astype(float) - g_aligned.astype(float))
            + np.abs(target_channels[2].astype(float) - r_aligned.astype(float))
        )
        return np.mean(diff)

    # Initiële verschuivingen vinden
    initial_shifts = [0, 0, 0, 0, 0, 0]
    res = minimize(
        calculate_alignment_error,
        initial_shifts,
        args=([b1, g1, r1], [b2, g2, r2]),
        method="Nelder-Mead",
    )
    b_dx, b_dy, g_dx, g_dy, r_dx, r_dy = res.x

    # Bereken transformatiematrices
    b_matrix = np.float32([[1, 0, b_dx], [0, 1, b_dy]])
    g_matrix = np.float32([[1, 0, g_dx], [0, 1, g_dy]])
    r_matrix = np.float32([[1, 0, r_dx], [0, 1, r_dy]])

    # Transformeer kleurkanalen
    b_aligned = cv2.warpAffine(
        b2,
        b_matrix,
        (prev_frame.shape[1], prev_frame.shape[0]),
        borderMode=cv2.BORDER_REPLICATE,
    )
    g_aligned = cv2.warpAffine(
        g2,
        g_matrix,
        (prev_frame.shape[1], prev_frame.shape[0]),
        borderMode=cv2.BORDER_REPLICATE,
    )
    r_aligned = cv2.warpAffine(
        r2,
        r_matrix,
        (prev_frame.shape[1], prev_frame.shape[0]),
        borderMode=cv2.BORDER_REPLICATE,
    )

    return cv2.merge([b_aligned, g_aligned, r_aligned])


def verwijder_lijnen(frame: MatLike):
    if len(frame.shape) != 2:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    gray = cv2.bitwise_not(gray)
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2
    )

    vertical = np.copy(bw)

    rows = vertical.shape[0]
    vertical_size = rows // 30

    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (3, vertical_size))

    vertical = cv2.erode(vertical, verticalStructure, iterations=5)
    vertical = cv2.dilate(vertical, verticalStructure, iterations=5)

    frame = cv2.inpaint(frame, vertical, 15, cv2.INPAINT_TELEA)

    return frame


def stabiliseer_en_mediaan_frame(frame: MatLike, enable: Enablers):
    """
    Average execution time: 2.6s - 2.7s
    """
    # If this is the first frame in the sequence, initialize frames
    if not hasattr(stabiliseer_en_mediaan_frame, "frames"):
        stabiliseer_en_mediaan_frame.frames = [frame] * 4

    frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 17)

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    frame = cv2.filter2D(frame, -1, kernel)

    if enable.remove_vertical_lines:
        frame = verwijder_lijnen(frame)
        b, g, r = cv2.split(frame)
        b = ndimage.median_filter(b, (1, 5))
        g = ndimage.median_filter(g, (1, 5))
        r = ndimage.median_filter(r, (1, 5))
        frame = cv2.merge((b, g, r))

    if enable.stabilize_colors:
        frame = align_and_stabilize_frame(
            stabiliseer_en_mediaan_frame.frames[-1], frame
        )

    if enable.stabilize:
        frame, _ = stabiliseer_frames(
            stabiliseer_en_mediaan_frame.frames[-1],
            frame,
            enable.stabilize_limit,
            enable.stabilize_only_x,
        )

    stabiliseer_en_mediaan_frame.frames = stabiliseer_en_mediaan_frame.frames[1:] + [
        frame
    ]

    # Calculate median of last 4 frames
    stacked_frames = np.stack(stabiliseer_en_mediaan_frame.frames, axis=-1)
    return np.median(stacked_frames, axis=-1).astype(np.uint8)


def crop_video_frame(frame, top=50, bottom=10, left=20, right=55):
    cropped_frame = frame[top : frame.shape[0] - bottom, left : frame.shape[1] - right]
    return cropped_frame


def evaluate_frames(frame: MatLike, frameOrig: MatLike):
    mse = metrics.mean_squared_error(frameOrig, frame)
    psnr = metrics.peak_signal_noise_ratio(frameOrig, frame)
    ssim = metrics.structural_similarity(frameOrig, frame, channel_axis=-1)
    return mse, psnr, ssim


def process_video(
    input_path: str,
    original: str,
    output_path: str,
    color_params: ColorParams,
    enable: Enablers,
):
    # Open the video files
    check_if_files_exist([input_path, original])
    cap = cv2.VideoCapture(input_path)
    capOrig = cv2.VideoCapture(original)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(
        f"-- Processing video: '{input_path}' with {frame_count} frames of {frame_width}x{frame_height} at {fps} fps"
    )

    # Create VideoWriter object
    fourcc = cv2.VideoWriter.fourcc("m", "p", "4", "v")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    if enable.bijsnijden:
        out = cv2.VideoWriter(
            output_path, fourcc, fps, (frame_width - 75, frame_height - 60)
        )

    eval_frame = 0
    evaluations = []

    for _ in tqdm(range(frame_count), desc="Progress"):
        # frame, frameOrig = threadpool.map(lambda e: e.read()[1], [cap, capOrig]) # not faster
        cap_ok, frame = cap.read()
        capOrig_ok, frameOrig = capOrig.read()

        if not cap_ok or not capOrig_ok:
            print("not cap_ok or not capOrig_ok")

        # -- Process frame --
        if enable.kleurrek:
            frame = optimaliseer_kleurrek(frame)

        frameOut = color_adjust(frame, frameOrig, color_params, enable.show_color_steps)
        frameOut = stabiliseer_en_mediaan_frame(frameOut, enable)

        if enable.bijsnijden:
            frameOut = crop_video_frame(frameOut)

        # -- Output frame --
        out.write(frameOut)

        # -- Evaluate every 10th frame --
        if enable.evaluate and (eval_frame % 10 == 0):
            evaluations.append(evaluate_frames(frame, frameOrig))
        eval_frame += 1

        if enable.show_processed_frame:
            cv2.imshow("Processing Video", frameOut)
        if enable.show_color_steps or cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print(f"Writen video to:", output_path)
    # Leeg de frame buffer voor de nieuwe video
    if hasattr(stabiliseer_en_mediaan_frame, "frames"):
        delattr(stabiliseer_en_mediaan_frame, "frames")

    if evaluations:
        mean_mse = np.mean([e[0] for e in evaluations])
        mean_psnr = np.mean([e[1] for e in evaluations])
        mean_ssim = np.mean([e[2] for e in evaluations])
        print(f"MSE={mean_mse}, PSNR={mean_psnr}, SSIM={mean_ssim}")

    # Release everything
    cap.release()
    capOrig.release()
    out.release()
    cv2.destroyAllWindows()
    return VideoFileClip(output_path)


def process_audio_and_video(
    input_path: str,
    output_filename: str,
    input_path_original: str = None,
    color_params: ColorParams = noEffectColor,
    enablers: Enablers = allOff,
    notch_filters: list[NotchFilter] = [],
    butterworth_filters: list[ButterworthFilters] = [],
    reduce_noise_filters: list = [],
    amplification_factor=1.0,
):
    processed_video = process_video(
        input_path,
        input_path_original if input_path_original else input_path,
        "output-temp/" + output_filename + ".mp4",
        color_params,
        enablers,
    )
    processed_audio = process_audio(
        input_path,
        "output-temp/" + output_filename + ".wav",
        input_path_original,
        notch_filters,
        butterworth_filters,
        reduce_noise_filters,
        amplification_factor,
        enablers.debug_audio,
    )
    combine_audio_with_video(
        processed_audio, processed_video, "output-final/" + output_filename + ".mp4"
    )
    print()


def main():
    # -- Startup --
    start_time = time.time()
    timestamp = time.strftime("%d-%m-%Y_%H%M%S")

    # -- Configuration --
    # - Color parameters -
    obamaColor = ColorParams(
        3, 1080, 0.005, 0.005, 1 / 3, 2.5, 2.1, 0, 190, 140, 20, 1, 0, 1
    )
    femaleColor = ColorParams(3, 1080, 0, 0, 1 / 2, 2.5, 2.1, 30, 190, 140, 40, 1, 0, 1)
    robinColor = ColorParams(3, 1080, 0, 0, 1 / 2, 2.5, 2.1, 30, 190, 140, 50, 1, 10, 1)
    archivePresident = ColorParams(3, 1080, 0, 0, 1 / 2, 1, 1, 30, 0, 0, -15, 1, -5, 1)
    archiveBreakfast = ColorParams(
        3, 1080 // 2, 0, 0, 2 / 3, 1, 1, 5, 0, 0, -20, 1, 5, 1
    )

    # - Other parameters -
    # Algemeen
    edit_no_show = Enablers(kleurrek=True, stabilize=True, evaluate=True)
    edit_with_show = Enablers(
        kleurrek=True, show_processed_frame=True, stabilize=True, evaluate=True
    )
    # Video specifiek
    edit_no_show_obama = Enablers(
        kleurrek=True,
        stabilize=True,
        stabilize_limit=True,
        evaluate=True,
        bijsnijden=True,
    )
    edit_no_show_female = Enablers(kleurrek=True, evaluate=True)
    edit_no_show_henry = Enablers(
        kleurrek=True,
        stabilize=True,
        stabilize_only_x=True,
        stabilize_limit=True,
        evaluate=True,
        bijsnijden=True,
    )
    edit_no_show_jasmine = Enablers(kleurrek=True, evaluate=True)
    edit_no_show_robin = Enablers(kleurrek=True, evaluate=True)
    removeLines = Enablers(remove_vertical_lines=True)

    # -- Video Processing --
    # - Degraded videos -
    process_audio_and_video(
        "../DegradedVideos/archive_2017-01-07_President_Obama's_Weekly_Address.mp4",
        f"output_obama-{timestamp}",
        "../SourceVideos/2017-01-07_President_Obama's_Weekly_Address.mp4",
        color_params=obamaColor,
        enablers=edit_no_show_obama,
        notch_filters=[NotchFilter(100, 30, 2)],
        butterworth_filters=[ButterworthFilters("lowpass", 5500, 5)],
        reduce_noise_filters=[ReduceNoiseFilters(False, 2048, 1)],
        amplification_factor=2.0,
    )
    
    process_audio_and_video(
        "../DegradedVideos/archive_20240709_female_common_yellowthroat_with_caterpillar_canoe_meadows.mp4",
        f"output_yellowthroat-{timestamp}",
        "../SourceVideos/20240709_female_common_yellowthroat_with_caterpillar_canoe_meadows.mp4",
        color_params=femaleColor,
        enablers=edit_no_show_female,
        notch_filters=[NotchFilter(100, 1, 2)],
        reduce_noise_filters=[ReduceNoiseFilters(False, 2048 * 4, 1)],
        amplification_factor=1.5,
    )

    process_audio_and_video(
        "../DegradedVideos/archive_Henry_Purcell__Music_For_a_While__-_Les_Arts_Florissants,_William_Christie.mp4",
        f"output_arts_florissants-{timestamp}",
        "../SourceVideos/Henry_Purcell__Music_For_a_While__-_Les_Arts_Florissants,_William_Christie.mp4",
        color_params=obamaColor,
        enablers=edit_no_show_henry,
        notch_filters=[NotchFilter(100, 1, 2)],
        butterworth_filters=[ButterworthFilters("lowpass", 10000, 5)],
        reduce_noise_filters=[ReduceNoiseFilters(True, 2048 * 4, 1)],
        amplification_factor=2.0,
    )
    
    process_audio_and_video(
        "../DegradedVideos/archive_Jasmine_Rae_-_Heartbeat_(Official_Music_Video).mp4",
        f"output_heartbeat-{timestamp}",
        "../SourceVideos/Jasmine_Rae_-_Heartbeat_(Official_Music_Video).mp4",
        color_params=obamaColor,
        enablers=edit_no_show_jasmine,
        notch_filters=[NotchFilter(100, 1, 2)],
        butterworth_filters=[ButterworthFilters("lowpass", 5500, 5)],
        reduce_noise_filters=[ReduceNoiseFilters(False, 2048 * 4, 1)],
        amplification_factor=1.0,
    )

    process_audio_and_video(
        "../DegradedVideos/archive_Robin_Singing_video.mp4",
        f"output_robin-{timestamp}",
        "../SourceVideos/Robin_Singing_video.mp4",
        color_params=robinColor,
        enablers=edit_no_show_robin,
        notch_filters=[NotchFilter(100, 1, 2)],
        butterworth_filters=[ButterworthFilters("lowpass", 5500, 5)],
        reduce_noise_filters=[ReduceNoiseFilters(False, 2048 * 4, 1)],
        amplification_factor=1.0,
    )

    # - Archive videos -
    process_audio_and_video(
        "../ArchiveVideos/Apollo_11_Landing_-_first_steps_on_the_moon.mp4",
        f"output_apollo-{timestamp}",
        color_params=noEffectColor,
        enablers=allOff,
        notch_filters=[
            NotchFilter(190, 1, 2),
            NotchFilter(110, 1, 2),
            NotchFilter(50, 1, 2),
        ],
        reduce_noise_filters=[ReduceNoiseFilters(False, 2048 * 2, 1)],
        amplification_factor=1.0,
    )

    process_audio_and_video(
        "..\ArchiveVideos\Breakfast-at-tiffany-s-official®-trailer-hd.mp4",
        f"output_tiffany-{timestamp}",
        color_params=archiveBreakfast,
        amplification_factor=1.0,
        enablers=allOff,
    )

    process_audio_and_video(
        "..\ArchiveVideos\Edison_speech,_1920s.mp4",
        f"output_edison-{timestamp}",
        color_params=noEffectColor,
        enablers=allOff,
        notch_filters=[NotchFilter(100, 1, 1)],
        butterworth_filters=[ButterworthFilters("lowpass", 5000, 7)],
        reduce_noise_filters=[ReduceNoiseFilters(False, 2048, 1)],
        amplification_factor=1.0,
    )

    process_audio_and_video(
        "..\ArchiveVideos\President_Kennedy_speech_on_the_space_effort_at_Rice_University,_September_12,_1962.mp4",
        f"output_kennedy-{timestamp}",
        color_params=archivePresident,
        enablers=allOff,
        notch_filters=[NotchFilter(50, 1, 1)],
        amplification_factor=1.0,
    )

    process_audio_and_video(
        "..\ArchiveVideos\The_Dream_of_Kings.mp4",
        f"output_king-{timestamp}",
        color_params=noEffectColor,
        enablers=removeLines,
        butterworth_filters=[ButterworthFilters("lowpass", 5500, 7)],
        reduce_noise_filters=[ReduceNoiseFilters(False, 2048, 1)],
        amplification_factor=2.0,
    )

    # -- Shutdown --
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Video processing completed in {execution_time:.2f} seconds")


if __name__ == "__main__":
    main()
