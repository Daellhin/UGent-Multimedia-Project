import math
import random
import cv2
import numpy as np
import scipy
from PIL.ImageChops import multiply
from matplotlib import pyplot as plt
from scipy.io import wavfile
import moviepy

def cumulative_histogram(image):
    hist = cv2.calcHist([image],[0],None,[256],[0,256])
    cumhist = np.cumsum(hist).astype(np.float64)
    cumhist = cumhist/cumhist[-1]
    return cumhist

def show_histogram(image, image2, nameImg="", nameImg2=""):
    fig = plt.figure(figsize=(20,15))
    ax = fig.add_subplot(321)
    ax.set_title(nameImg)
    ax0 = fig.add_subplot(322)
    ax0.set_title(nameImg2)
    ax1 = fig.add_subplot(323)
    ax1.set_title("Histogram of "+nameImg)
    ax2 = fig.add_subplot(324, sharex=ax1)
    ax2.set_title("Histogram of "+nameImg2)
    ax3 = fig.add_subplot(325, sharex=ax1)
    ax3.set_title("Cumulative Histogram of "+nameImg)
    ax4 = fig.add_subplot(326, sharex=ax1)
    ax4.set_title("Cumulative Histogram of "+nameImg2)
    ax1.set_xlim([0,255])
    ax.imshow(image, cmap=plt.get_cmap("gray"))
    ax0.imshow(image2, cmap=plt.get_cmap("gray"))
    ax1.plot(cv2.calcHist([image],[0],None,[256],[0,256]))
    ax2.plot(cv2.calcHist([image2],[0],None,[256],[0,256]))
    ax3.plot(cumulative_histogram(image))
    ax4.plot(cumulative_histogram(image2))
    plt.show()

def magnitude_spectrum(fshift):
    magnitude_spectrum = np.log(np.abs(fshift) + 1)
    magnitude_spectrum -= np.min(magnitude_spectrum)
    magnitude_spectrum *= 255. / np.max(magnitude_spectrum)
    return magnitude_spectrum

def show_spectrum(image, original_image, figure_name):
    fft_image = np.fft.fft2(image)
    fft_image = np.fft.fftshift(fft_image)
    fft_original = np.fft.fft2(original_image)
    fft_original = np.fft.fftshift(fft_original)
    fig = plt.figure()
    fig.suptitle(figure_name, fontsize=14)
    ax1 = fig.add_subplot(221)
    ax1.set_title(figure_name+" before")
    ax2 = fig.add_subplot(222)
    ax2.set_title("FFT")
    ax3 = fig.add_subplot(223)
    ax3.set_title("FFT original")
    ax4 = fig.add_subplot(224)
    ax4.set_title(figure_name+" original")
    ax1.imshow(image, cmap = "gray")
    ax2.imshow(magnitude_spectrum(fft_image), cmap = "gray")
    ax3.imshow(magnitude_spectrum(fft_original), cmap ="gray")
    ax4.imshow(original_image, cmap ="gray")
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    ax4.axis('off')
    plt.tight_layout()
    plt.show()

def butterworth_filter(shape,n,D0):
    H = np.zeros(shape)
    rows, cols = shape
    center_row, center_col = rows / 2, cols / 2

    for u in range(rows):
        for v in range(cols):
            D = np.sqrt((u - center_row) ** 2 + (v - center_col) ** 2)
            H[u, v] = 1 / (1 + (-D / D0) ** (2 * n))
    return H

def histogram_matching(inframe,orframe,show=False,type=""):
    if show:
        show_histogram(orframe, inframe, type + " original", type + " before")
    ref = cumulative_histogram(orframe)
    low = cumulative_histogram(inframe)
    low_normalized = low / low.max()
    ref_normalized = ref / ref.max()

    lookup_table = np.zeros(256, dtype=np.uint8)
    j = 1
    for i in range(256):
        while j < 256 and ref_normalized[j] <= low_normalized[i]:
            j += 1
        lookup_table[i] = j - 1

    outframe = cv2.LUT(inframe, lookup_table)

    if show:
        show_histogram(orframe, outframe, type + " original", type + " matched")
    return outframe

def process_frame(frame, frameOrig,show_steps=False):
    # BGR modifier
    yuv = cv2.cvtColor(frame,cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv)
    yuvor = cv2.cvtColor(frameOrig, cv2.COLOR_BGR2YUV)
    yo, uo, vo = cv2.split(yuvor)

    if show_steps:
        cv2.imwrite("output/start_Frame.jpg",frame)
        #show_histogram(y, yo, "Y", "Y Original")
        #show_spectrum(y, yo, "Y")
        #show_histogram(u, uo, "U", "U original")
        #show_spectrum(u, uo, "U")
        #show_histogram(v, vo, "V", "V Original")
        #show_spectrum(v, vo, "V")

    #y = cv2.add(y,10)
    u = cv2.subtract(u,77)
    v = cv2.subtract(v,77)
    y = cv2.multiply(y,0.90)
    u = cv2.multiply(u,2.5)
    v = cv2.multiply(v,2.5)
    #y = scipy.ndimage.gaussian_filter(y, 1.5)
    #u = scipy.ndimage.gaussian_filter(u, 1.2)
    #v = scipy.ndimage.gaussian_filter(v, 1.2)

    rows, cols = v.shape
    kernel_x = cv2.getGaussianKernel(cols, 2*cols)
    kernel_y = cv2.getGaussianKernel(rows, 2*rows)
    kernel = kernel_y * kernel_x.T
    mask = (1 - kernel / np.linalg.norm(kernel))
    mask = cv2.normalize(mask, None, 0, 4, cv2.NORM_MINMAX).astype(np.uint8)
    if show_steps:
        plt.imshow(mask)
        plt.show()
    y = cv2.add(y, 4*mask)
    u = cv2.add(u, mask)
    v = cv2.add(v, mask)

    y = np.clip(y,0,255)
    u = np.clip(u,0,255)
    v = np.clip(v,0,255)
    #y = cv2.normalize(y, None, 0, 255, cv2.NORM_MINMAX)
    #u = cv2.normalize(u,None,0,255,cv2.NORM_MINMAX)
    #v = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)

    if show_steps:
        cv2.imwrite("output/YUV-edit_Frame.jpg",frame)
        show_histogram(y, yo, "Y edit", "Y Original")
        #show_spectrum(y, yo, "Y")
        show_histogram(u, uo, "U edit", "U original")
        #show_spectrum(u, uo, "U")
        show_histogram(v, vo, "V edit", "V Original")
        #show_spectrum(v, vo, "V")


    frame = cv2.merge((y, u, v))

    # BGR modifier
    frame = cv2.cvtColor(frame,cv2.COLOR_YUV2BGR)
    #b, g, r = cv2.split(frame)
    #bo, go, ro = cv2.split(frameOrig)

    #r = np.clip(r, 0, 255)
    #g = np.clip(g, 0, 255)
    #b = np.clip(b, 0, 255)
    """
    if show_steps:
        cv2.imwrite("output/start_Frame.jpg",frame)
        show_histogram(r, ro, "Red", "Red Original")
        show_spectrum(r, ro, "Red")
        show_histogram(g, go, "Green", "Green Original")
        show_spectrum(g, go, "Green")
        show_histogram(b, bo, "Blue", "Blue Original")
        show_spectrum(b, bo, "Blue")
    """
    """fft_r = np.fft.fft2(r)
    fft_r = np.fft.fftshift(fft_r)
    fft_r_filter = fft_r * butterworth_filter(r.shape, 5, 300)
    ifft_r = np.fft.ifftshift(fft_r_filter)
    ifft_r = np.fft.ifft2(ifft_r)
    r = ifft_r.real
    fft_g = np.fft.fft2(g)
    fft_g = np.fft.fftshift(fft_g)
    fft_g_filter = fft_g * butterworth_filter(r.shape, 5, 300)
    ifft_g = np.fft.ifftshift(fft_g_filter)
    ifft_g = np.fft.ifft2(ifft_g)
    g = ifft_g.real
    fft_b = np.fft.fft2(b)
    fft_b = np.fft.fftshift(fft_b)
    fft_b_filter = fft_b * butterworth_filter(r.shape, 5, 300)
    ifft_b = np.fft.ifftshift(fft_b_filter)
    ifft_b = np.fft.ifft2(ifft_b)
    b = ifft_b.real
    if show_steps:
        show_spectrum(r, ro, "Red")
        show_spectrum(g, go, "Green")
        show_spectrum(b, bo, "Blue")"""

    #r = scipy.ndimage.gaussian_filter(r, 2)
    #g = scipy.ndimage.gaussian_filter(g, 2.5)
    #b = scipy.ndimage.gaussian_filter(b, 2)

    #r = cv2.multiply(r,0.80)
    #g = cv2.multiply(g,0.70)
    #b = cv2.multiply(b,0.6)

    #r = np.clip(r,0,255)
    #g = np.clip(g, 0, 255)
    #b = np.clip(b, 0, 255)
    #r = cv2.normalize(r, None, 0, 255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    #g = cv2.normalize(g,None,0,255,norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    #b = cv2.normalize(b,None,0,255,norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    #frame = cv2.merge((b,g,r))

    #HSV modifiers
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    hsvOrig = cv2.cvtColor(frameOrig, cv2.COLOR_BGR2HSV)
    ho, so, vo = cv2.split(hsvOrig)
    
    if show_steps:
        cv2.imwrite("output/edit_frame.jpg",frame)
        show_histogram(h, ho, "Hue", "Hue Original")
        #show_spectrum(h,ho,"Hue")
        show_histogram(v, vo, "Value", "Value Original")
        #show_spectrum(v,vo,"Value")
        show_histogram(s, so, "Saturation", "Saturation Original")
        #show_spectrum(s,so,"Saturation")

    #h = cv2.multiply(h,0.99)
    v = cv2.multiply(v, 0.90)
    s = cv2.add(s, 10)
    s = cv2.multiply(s,1.20)

    #h = scipy.ndimage.median_filter(h,(5,5))
    #v = scipy.ndimage.maximum_filter(v, (3,3))
    #s = scipy.ndimage.maximum_filter(s, (3,3))
    #h = scipy.ndimage.gaussian_filter(h,1.1)
    s = scipy.ndimage.gaussian_filter(s,1.5)
    v = scipy.ndimage.gaussian_filter(v,1.5)

    #v = cv2.normalize(v,None,0,255,norm_type=cv2.NORM_MINMAX)
    #s = cv2.normalize(s,None,0,255,norm_type=cv2.NORM_MINMAX)
    v = np.clip(v, 0, 255)
    s = np.clip(s, 0, 255)

    final_hsv = cv2.merge((h, s, v))
    frame = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    
    if show_steps:
        show_histogram(h, ho, "Hue", "Hue Original")
        show_histogram(v, vo, "Value", "Value Original")
        show_histogram(s, so, "Saturation", "Saturation Original")

    return frame

def process_video(input_path,original, output_path, show_steps=False, show_processed_frame=True):
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

    while cap.isOpened() and capOrig.isOpened():
        ret, frame = cap.read()
        ret, frameOrig = capOrig.read()
        if not ret:
            break

        frame = process_frame(frame,frameOrig,show_steps)
        # Write the processed frame
        out.write(frame)

        if show_processed_frame:
            cv2.imshow('Processing Video', frame)
        if show_steps or cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    print(moviepy.__version__)
    """video_clip = VideoFileClip("../DegradedVideos/archive_2017-01-07_President_Obama's_Weekly_Address.mp4")
    W,H = video_clip.size
    print(W,H)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile("output/original_audio.mp3")
    audio_clip = AudioFileClip("output/edit_audio.mp3")
    video_clip = video_clip.set_audio(audio_clip)
    video_clip.write_videofile("output/original_video.mp4")
    audio_clip.close()
    video_clip.close()"""

    process_video("../DegradedVideos/archive_2017-01-07_President_Obama's_Weekly_Address.mp4",
                  "../SourceVideos/2017-01-07_President_Obama's_Weekly_Address.mp4",
                  "output/2017-01-07_President_Obama's_Weekly_Address.mp4")
    process_video("../DegradedVideos/archive_20240709_female_common_yellowthroat_with_caterpillar_canoe_meadows.mp4",
                  "../SourceVideos/20240709_female_common_yellowthroat_with_caterpillar_canoe_meadows.mp4",
                  "output/20240709_female_common_yellowthroat_with_caterpillar_canoe_meadows.mp4")
    process_video("../DegradedVideos/archive_Henry_Purcell__Music_For_a_While__-_Les_Arts_Florissants,_William_Christie.mp4",
                  "../SourceVideos/Henry_Purcell__Music_For_a_While__-_Les_Arts_Florissants,_William_Christie.mp4",
                  "output/Henry_Purcell__Music_For_a_While__-_Les_Arts_Florissants,_William_Christie.mp4")
    process_video("../DegradedVideos/archive_Robin_Singing_video.mp4",
                  "../SourceVideos/Robin_Singing_video.mp4",
                  "output/Robin_Singing_video.mp4")
    process_video("../DegradedVideos/archive_Jasmine_Rae_-_Heartbeat_(Official_Music_Video).mp4",
                  "../SourceVideos/Jasmine_Rae_-_Heartbeat_(Official_Music_Video).mp4",
                  "output/Jasmine_Rae_-_Heartbeat_(Official_Music_Video).mp4")

if __name__ == '__main__':
    main()