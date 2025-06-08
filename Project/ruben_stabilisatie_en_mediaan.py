import math
import random
from pydoc import importfile
from signal import signal
from scipy.optimize import minimize

import cv2
import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy.io import wavfile
import moviepy
from visualisations import *
from utils import printProgressBar

def create_gaussian_kernel(size=15, sigma=3):
    """Create a 2D Gaussian kernel."""
    x = np.linspace(-size // 2, size // 2, size)
    y = np.linspace(-size // 2, size // 2, size)
    x, y = np.meshgrid(x, y)
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel / kernel.sum()

def process_frame(frame, frameOrig, show_steps=False):
    # YUV modifier - kringverzwakking
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv)
    yuv_or = cv2.cvtColor(frameOrig, cv2.COLOR_BGR2YUV)
    yo, uo, vo = cv2.split(yuv_or)

    if show_steps:
        cv2.imwrite("output/start_Frame.jpg", frame)
        # show_histogram(y, yo, "Y", "Y Original")
        # show_spectrum(y, yo, "Y")
        # show_histogram(u, uo, "U", "U original")
        # show_spectrum(u, uo, "U")
        # show_histogram(v, vo, "V", "V Original")
        # show_spectrum(v, vo, "V")

    # y = scipy.ndimage.median_filter(y, (3,3))
    y = cv2.blur(y, (5, 5))

    rows, cols = v.shape
    kernel_x = cv2.getGaussianKernel(cols, 1080/2)
    kernel_y = cv2.getGaussianKernel(rows, 1080/2)
    kernel = kernel_y * kernel_x.T
    mask = 1 - kernel / np.linalg.norm(kernel)
    mask = cv2.normalize(mask, None, 0, 1, cv2.NORM_MINMAX)
    if show_steps:
        plt.imshow(mask)
        plt.show()
    y = cv2.multiply(y.astype(np.float64), 3 / 4 + 1 / 2 * mask)
    u = cv2.multiply(u.astype(np.float64), 1 + 0.01 * mask)
    v = cv2.multiply(v.astype(np.float64), 1 + 0.01 * mask)
    u = cv2.multiply(u, 2.5)
    u = cv2.subtract(u, 190)
    v = cv2.multiply(v, 2.1)
    v = cv2.subtract(v, 140)

    # u = scipy.ndimage.gaussian_filter(u,1.2)
    # v = scipy.ndimage.gaussian_filter(v, 1.2)
    y = np.clip(y, 0, 255).astype(np.uint8)
    u = np.clip(u, 0, 255).astype(np.uint8)
    v = np.clip(v, 0, 255).astype(np.uint8)
    frame = cv2.merge((y, u, v))

    frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR)
    if show_steps:
        cv2.imwrite("output/YUV-edit_Frame.jpg", frame)
        show_histogram(y, yo, "Y edit", "Y Original")
        # show_spectrum(y, yo, "Y")
        show_histogram(u, uo, "U edit", "U original")
        # show_spectrum(u, uo, "U")
        show_histogram(v, vo, "V edit", "V Original")
        # show_spectrum(v, vo, "V")

    # BGR modifier
    b, g, r = cv2.split(frame)
    bo, go, ro = cv2.split(frameOrig)

    if show_steps:
        show_histogram(r, ro, "Red", "Red Original")
        show_spectrum(r, ro, "Red")
        show_histogram(g, go, "Green", "Green Original")
        show_spectrum(g, go, "Green")
        show_histogram(b, bo, "Blue", "Blue Original")
        show_spectrum(b, bo, "Blue")

    # r = scipy.ndimage.gaussian_filter(r, 2)
    # g = scipy.ndimage.gaussian_filter(g, 2.5)
    # b = scipy.ndimage.gaussian_filter(b, 2)
    # r = cv2.multiply(r,0.80)
    # g = cv2.multiply(g,0.70)
    # b = cv2.multiply(b,0.6)

    r = np.clip(r, 0, 255).astype(np.uint8)
    g = np.clip(g, 0, 255).astype(np.uint8)
    b = np.clip(b, 0, 255).astype(np.uint8)
    frame = cv2.merge((b, g, r))

    # HSV modifiers
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    hsvOrig = cv2.cvtColor(frameOrig, cv2.COLOR_BGR2HSV)
    ho, so, vo = cv2.split(hsvOrig)

    if show_steps:
        cv2.imwrite("output/BGR-edit_frame.jpg", frame)
        # show_histogram(h, ho, "Hue", "Hue Original")
        # show_spectrum(h,ho,"Hue")
        # show_histogram(v, vo, "Value", "Value Original")
        # show_spectrum(v,vo,"Value")
        # show_histogram(s, so, "Saturation", "Saturation Original")
        # show_spectrum(s,so,"Saturation")

    # h = scipy.ndimage.median_filter(h,(1,3))
    # s = scipy.ndimage.median_filter(s, (5,5))
    # v = scipy.ndimage.median_filter(v,(3,3))

    # h = cv2.multiply(h,0.99)
    # v = cv2.multiply(v, 0.90)
    # s = cv2.multiply(s,1.75)
    s = cv2.add(s, 20)

    # s = scipy.ndimage.gaussian_filter(s, 4)
    h = np.clip(h, 0, 255).astype(np.uint8)
    v = np.clip(v, 0, 255).astype(np.uint8)
    s = np.clip(s, 0, 255).astype(np.uint8)
    final_hsv = cv2.merge((h, s, v))
    frame = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    if show_steps:
        show_histogram(h, ho, "Hue", "Hue Original")
        show_histogram(v, vo, "Value", "Value Original")
        show_histogram(s, so, "Saturation", "Saturation Original")


    def remove_black_borders_crop(aligned_img):
        # Zoek de eerste niet-zwarte rijen en kolommen
        non_black_rows = np.any(aligned_img != 0, axis=1)
        non_black_cols = np.any(aligned_img != 0, axis=0)

        # Bepaal de grenzen
        row_min, row_max = np.where(non_black_rows)[0][[0, -1]]
        col_min, col_max = np.where(non_black_cols)[0][[0, -1]]

        # Bijsnijden
        cropped_img = aligned_img[row_min:row_max + 1, col_min:col_max + 1]
        return cropped_img

    def stabiliseer_frames(img1, img2):
        # Convert images to grayscale for feature detection
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Use ORB for feature detection and matching
        orb = cv2.ORB_create()

        # Detect keypoints and compute descriptors
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
        translation_matrix, _ = cv2.estimateAffinePartial2D(dst_pts, src_pts)

        # Apply translation to the color image
        #aligned_img2 = cv2.warpAffine(img2, translation_matrix, (img1.shape[1], img1.shape[0]))

        aligned_img2 = cv2.warpAffine(
            img2,
            translation_matrix,
            (img1.shape[1], img1.shape[0]),
            borderMode=cv2.BORDER_REPLICATE  # Herhaalt de randpixels
        )

        return aligned_img2, translation_matrix

    def stabiliseer_frames_met_segmentatie(img1, img2):
        # Converteer naar grijswaarden voor feature detectie
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Gebruik een voorgrond segmentatie methode (bijv. GrabCut)
        mask1 = np.zeros(img1.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        # Ruwe rechthoek rond het voorgrond object (pas aan aan uw specifieke gebruik)
        rect = (50, 50, img1.shape[1] - 150, img1.shape[0] - 150)
        cv2.grabCut(img1, mask1, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

        # Maak een mask van alleen de achtergrond
        mask = np.where((mask1 == 2) | (mask1 == 0), 0, 1).astype('uint8')

        # Gebruik ORB alleen op de achtergrondpixels
        orb = cv2.ORB_create()

        # Maak maskers voor keypoint detectie
        gray1_masked = cv2.bitwise_and(gray1, gray1, mask=mask)
        gray2_masked = cv2.bitwise_and(gray2, gray2, mask=mask)

        # Detecteer keypoints op de gemaskeerde afbeeldingen
        kp1, des1 = orb.detectAndCompute(gray1_masked, mask)
        kp2, des2 = orb.detectAndCompute(gray2_masked, mask)

        # Rest van de matching logica blijft hetzelfde als in uw oorspronkelijke code
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:10]

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Schat translatiematrix
        translation_matrix, _ = cv2.estimateAffinePartial2D(dst_pts, src_pts)

        # Pas translatie toe met randpixel replicatie
        aligned_img2 = cv2.warpAffine(
            img2,
            translation_matrix,
            (img1.shape[1], img1.shape[0]),
            borderMode=cv2.BORDER_REPLICATE
        )

        return aligned_img2, translation_matrix

    def stabiliseer_frames_randen(img1, img2):
        # Convert images to grayscale for feature detection
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Select only the edges of the images
        height, width = gray1.shape
        left_edge1 = gray1[:, :width // 3]
        right_edge1 = gray1[:, 2 * width // 3:]
        edges1 = np.hstack((left_edge1, right_edge1))

        left_edge2 = gray2[:, :width // 3]
        right_edge2 = gray2[:, 2 * width // 3:]
        edges2 = np.hstack((left_edge2, right_edge2))

        # Use ORB for feature detection and matching
        orb = cv2.ORB_create()

        # Detect keypoints and compute descriptors
        kp1, des1 = orb.detectAndCompute(edges1, None)
        kp2, des2 = orb.detectAndCompute(edges2, None)

        # Create brute-force matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors
        matches = bf.match(des1, des2)

        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Select good matches
        good_matches = matches[:10]

        # Extract matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Estimate translation matrix
        translation_matrix, _ = cv2.estimateAffinePartial2D(dst_pts, src_pts)

        # Apply translation to the color image
        aligned_img2 = cv2.warpAffine(
            img2,
            translation_matrix,
            (img1.shape[1], img1.shape[0]),
            borderMode=cv2.BORDER_REPLICATE  # Herhaalt de randpixels
        )

        return aligned_img2, translation_matrix

    def stabiliseer_frames_per_kleurkanaal(img1, img2):
        # Split de afbeeldingen in afzonderlijke kleurkanalen
        b1, g1, r1 = cv2.split(img1)
        b2, g2, r2 = cv2.split(img2)

        # Functies voor feature detectie en matching per kleurkanaal
        def bereken_transformatie(channel1, channel2):
            # Convert channels to grayscale for feature detection
            gray1 = channel1
            gray2 = channel2

            # Use ORB for feature detection and matching
            orb = cv2.ORB_create()

            # Detect keypoints and compute descriptors
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
            translation_matrix, _ = cv2.estimateAffinePartial2D(dst_pts, src_pts)

            return translation_matrix

        # Bereken transformatiematrices voor elk kleurkanaal
        b_matrix = bereken_transformatie(b1, b2)
        g_matrix = bereken_transformatie(g1, g2)
        r_matrix = bereken_transformatie(r1, r2)

        # Pas transformaties toe op elk kleurkanaal
        b_aligned = cv2.warpAffine(
            b2,
            b_matrix,
            (img1.shape[1], img1.shape[0]),
            borderMode=cv2.BORDER_REPLICATE
        )

        g_aligned = cv2.warpAffine(
            g2,
            g_matrix,
            (img1.shape[1], img1.shape[0]),
            borderMode=cv2.BORDER_REPLICATE
        )

        r_aligned = cv2.warpAffine(
            r2,
            r_matrix,
            (img1.shape[1], img1.shape[0]),
            borderMode=cv2.BORDER_REPLICATE
        )

        # Recombineer de kleurkanalen met hun eigen unieke transformaties
        aligned_img2 = cv2.merge([b_aligned, g_aligned, r_aligned])

        return aligned_img2, (b_matrix, g_matrix, r_matrix)

    def align_color_channels(frame):
        b, g, r = cv2.split(frame)

        # Corrigeer de kleurverschuiving tussen de kanalen
        b_shift = np.array([1, 5])  # Bepaal de verschuiving voor het blauwe kanaal
        g_shift = np.array([5, 0])  # Geen verschuiving voor groen
        r_shift = np.array([5, 0])  # Bepaal de verschuiving voor rood

        # Transformeer de kanalen om ze uit te lijnen
        b_aligned = cv2.warpAffine(b, np.float32([[1, 0, b_shift[0]], [0, 1, b_shift[1]]]), frame.shape[1::-1])
        g_aligned = g
        r_aligned = cv2.warpAffine(r, np.float32([[1, 0, r_shift[0]], [0, 1, r_shift[1]]]), frame.shape[1::-1])

        # Combineer de uitgelijnd kanalen
        aligned_img = cv2.merge([b_aligned, g_aligned, r_aligned])
        return aligned_img

    def align_and_stabilize_frame(prev_frame, curr_frame):
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
            b_aligned = cv2.warpAffine(source_channels[0], np.float32([[1, 0, b_dx], [0, 1, b_dy]]),
                                       (target_channels[0].shape[1], target_channels[0].shape[0]))
            g_aligned = cv2.warpAffine(source_channels[1], np.float32([[1, 0, g_dx], [0, 1, g_dy]]),
                                       (target_channels[1].shape[1], target_channels[1].shape[0]))
            r_aligned = cv2.warpAffine(source_channels[2], np.float32([[1, 0, r_dx], [0, 1, r_dy]]),
                                       (target_channels[2].shape[1], target_channels[2].shape[0]))

            # Bereken verschil tussen gealigneerde en doel-kanalen
            diff = np.abs(target_channels[0].astype(float) - b_aligned.astype(float)) + \
                   np.abs(target_channels[1].astype(float) - g_aligned.astype(float)) + \
                   np.abs(target_channels[2].astype(float) - r_aligned.astype(float))
            return np.mean(diff)

        # Initiële verschuivingen vinden
        initial_shifts = [0, 0, 0, 0, 0, 0]
        res = minimize(calculate_alignment_error, initial_shifts, args=([b1, g1, r1], [b2, g2, r2]),
                       method='Nelder-Mead')
        b_dx, b_dy, g_dx, g_dy, r_dx, r_dy = res.x

        # Bereken transformatiematrices
        b_matrix = np.float32([[1, 0, b_dx], [0, 1, b_dy]])
        g_matrix = np.float32([[1, 0, g_dx], [0, 1, g_dy]])
        r_matrix = np.float32([[1, 0, r_dx], [0, 1, r_dy]])

        # Transformeer kleurkanalen
        b_aligned = cv2.warpAffine(b2, b_matrix, (prev_frame.shape[1], prev_frame.shape[0]),
                                   borderMode=cv2.BORDER_REPLICATE)
        g_aligned = cv2.warpAffine(g2, g_matrix, (prev_frame.shape[1], prev_frame.shape[0]),
                                   borderMode=cv2.BORDER_REPLICATE)
        r_aligned = cv2.warpAffine(r2, r_matrix, (prev_frame.shape[1], prev_frame.shape[0]),
                                   borderMode=cv2.BORDER_REPLICATE)

        # Recombineer kleurkanalen
        aligned_and_stabilized = cv2.merge([b_aligned, g_aligned, r_aligned])

        return aligned_and_stabilized

    def optimaliseer_kleuren(frame):
        # Splits de afbeelding in BGR-kanalen
        b, g, r = cv2.split(frame)

        # Definieer het centrum van de afbeelding (assumeer centraal)
        height, width = g.shape
        center_x, center_y = width // 2, height // 2

        # Functie om radiale verschuiving te berekenen
        def radial_shift_map(shape, scale_factor, power=1.0):
            """Maak een verschuivingskaart op basis van een radiale functie."""
            h, w = shape
            y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            dx = x - center_x
            dy = y - center_y
            r = np.sqrt(dx * 2 + dy * 2)  # Radiale afstand
            # Bereken de verschuiving
            shift_x = scale_factor * (dx / r) * (r ** power)
            shift_y = scale_factor * (dy / r) * (r ** power)
            shift_x[np.isnan(shift_x)] = 0  # Voorkom deling door 0
            shift_y[np.isnan(shift_y)] = 0
            return shift_x, shift_y

        # Corrigeer een kanaal
        def apply_radial_shift(channel, shift_x, shift_y):
            """Pas de verschuivingen toe op een kleurkanaal."""
            map_x, map_y = np.meshgrid(np.arange(channel.shape[1]), np.arange(channel.shape[0]))
            map_x = (map_x - shift_x).astype(np.float32)
            map_y = (map_y - shift_y).astype(np.float32)
            corrected = cv2.remap(channel, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            return corrected

        # Parameters voor verschuivingen
        scale_factor_red = 0.004  # Pas aan op basis van quiver-plot of experimenten
        scale_factor_blue = -0.003

        # Bereken verschuivingskaarten voor rode en blauwe kanalen
        shift_x_r, shift_y_r = radial_shift_map(g.shape, scale_factor_red)
        shift_x_b, shift_y_b = radial_shift_map(g.shape, scale_factor_blue)

        # Corrigeer de rode en blauwe kanalen
        aligned_r = apply_radial_shift(r, shift_x_r, shift_y_r)
        aligned_b = apply_radial_shift(b, shift_x_b, shift_y_b)

        # Combineer de gecorrigeerde kanalen
        aligned_image = cv2.merge((aligned_b, g, aligned_r))

        return aligned_image

    # Function to process frames with median filter
    def mediaan_filter_op_frame_basis(frame):
        # If this is the first frame in the sequence, initialize frames
        if not hasattr(process_frame, 'frames'):
            process_frame.frames = [frame] * 4

        #frame, translation_matrix = stabiliseer_frames(process_frame.frames[-1],frame)
        frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 17)

        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        frame = cv2.filter2D(frame, -1, kernel)

        #frame = align_color_channels(frame)

        #frame, translation_matrix = align_and_stabilize_frame(process_frame.frames[-1],frame)


        frame = align_and_stabilize_frame(process_frame.frames[-1],frame)
        frame, _ = stabiliseer_frames(process_frame.frames[-1],frame)

        process_frame.frames = process_frame.frames[1:] + [frame]

        # Calculate median of last 4 frames
        stacked_frames = np.stack(process_frame.frames, axis=-1)
        return np.median(stacked_frames, axis=-1).astype(np.uint8)

    #frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 12)

    frame = optimaliseer_kleuren(frame)

    frame = mediaan_filter_op_frame_basis(frame)

    return frame

def process_video(input_path, original, output_path, show_steps=False, show_processed_frame=True):
    # Open the video file
    cap = cv2.VideoCapture(input_path)
    capOrig = cv2.VideoCapture(original)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create VideoWriter object
    fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    total_iterations = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(fps)
    print(total_iterations)
    counter_iterations = 0
    printProgressBar(counter_iterations,total_iterations)

    while cap.isOpened() and capOrig.isOpened():
        ret, frame = cap.read()
        ret, frameOrig = capOrig.read()
        if not ret:
            break

        frame = process_frame(frame, frameOrig, show_steps)
        # Write the processed frame
        out.write(frame)

        printProgressBar(counter_iterations, total_iterations)
        counter_iterations += 1
        if show_processed_frame:
            cv2.imshow('Processing Video', frame)
        if show_steps or cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    """
    process_video("../DegradedVideos/archive_2017-01-07_President_Obama's_Weekly_Address.mp4",
                  "../SourceVideos/2017-01-07_President_Obama's_Weekly_Address.mp4",
                  "output/archive_2017-01-07_President_Obama's_Weekly_Address.mp4")

    process_video("../DegradedVideos/archive_20240709_female_common_yellowthroat_with_caterpillar_canoe_meadows.mp4",
                  "../SourceVideos/20240709_female_common_yellowthroat_with_caterpillar_canoe_meadows.mp4",
                  "output/20240709_female_common_yellowthroat_with_caterpillar_canoe_meadows.mp4",False)
    
    process_video("../DegradedVideos/archive_Henry_Purcell__Music_For_a_While__-_Les_Arts_Florissants,_William_Christie.mp4",
                  "../SourceVideos/Henry_Purcell__Music_For_a_While__-_Les_Arts_Florissants,_William_Christie.mp4",
                  "output/Henry_Purcell__Music_For_a_While__-_Les_Arts_Florissants,_William_Christie.mp4")
    
    process_video("../DegradedVideos/archive_Robin_Singing_video.mp4",
                  "../SourceVideos/Robin_Singing_video.mp4",
                  "output/Robin_Singing_video.mp4")
    
    process_video("../DegradedVideos/archive_Jasmine_Rae_-_Heartbeat_(Official_Music_Video).mp4",
                  "../SourceVideos/Jasmine_Rae_-_Heartbeat_(Official_Music_Video).mp4",
                  "output/Jasmine_Rae_-_Heartbeat_(Official_Music_Video).mp4")

    process_video("../ArchiveVideos/Apollo_11_Landing_-_first_steps_on_the_moon.mp4",
                  "../ArchiveVideos/Apollo_11_Landing_-_first_steps_on_the_moon.mp4",
                  "output/Apollo_11_Landing_-_first_steps_on_the_moon.mp4")

    process_video("..\ArchiveVideos\Breakfast-at-tiffany-s-official®-trailer-hd.mp4",
                  "..\ArchiveVideos\Breakfast-at-tiffany-s-official®-trailer-hd.mp4",
                  "output\Breakfast-at-tiffany-s-official®-trailer-hd.mp4")
    
    process_video("..\ArchiveVideos\Edison_speech,_1920s.mp4",
                  "..\ArchiveVideos\Edison_speech,_1920s.mp4",
                  "output\Edison_speech,_1920s.mp4")
    """
    process_video("..\ArchiveVideos\President_Kennedy_speech_on_the_space_effort_at_Rice_University,_September_12,_1962.mp4",
                  "..\ArchiveVideos\President_Kennedy_speech_on_the_space_effort_at_Rice_University,_September_12,_1962.mp4",
                  "output\President_Kennedy_speech_on_the_space_effort_at_Rice_University,_September_12,_1962.mp4")

    process_video("..\ArchiveVideos\The_Dream_of_Kings.mp4",
                  "..\ArchiveVideos\The_Dream_of_Kings.mp4",
                  "output\The_Dream_of_Kings.mp4")

if __name__ == '__main__':
    main()