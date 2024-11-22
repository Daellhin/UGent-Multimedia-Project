import math
import random
import cv2
import numpy as np
import scipy.ndimage
from matplotlib import pyplot as plt
from scipy.io import wavfile
import moviepy

def cumulative_histogram(image):
    hist = cv2.calcHist([image],[0],None,[256],[0,256])
    cumhist = np.cumsum(hist).astype(np.float64)
    cumhist = cumhist/cumhist[-1]
    return cumhist

def show_results(image, result_image):
    fig = plt.figure(figsize=(20,15))
    ax = fig.add_subplot(321)
    ax.set_title("Original Image")
    ax0 = fig.add_subplot(322)
    ax0.set_title("Image after Transformation")
    ax1 = fig.add_subplot(323)
    ax1.set_title("Histogram of Original")
    ax2 = fig.add_subplot(324, sharex=ax1)
    ax2.set_title("Histogram after Transformation")
    ax3 = fig.add_subplot(325, sharex=ax1)
    ax3.set_title("Cumulative Histogram of Original")
    ax4 = fig.add_subplot(326, sharex=ax1)
    ax4.set_title("Cumulative Histogram after Transformation")
    ax1.set_xlim([0,255])
    ax.imshow(image, cmap=plt.get_cmap("gray"))
    ax0.imshow(result_image, cmap=plt.get_cmap("gray"))
    ax1.plot(cv2.calcHist([image],[0],None,[256],[0,256]))
    ax2.plot(cv2.calcHist([result_image],[0],None,[256],[0,256]))
    ax3.plot(cumulative_histogram(image))
    ax4.plot(cumulative_histogram(result_image))
    plt.show()

def histogram_matching(inframe,orframe):
    show_results(orframe, inframe)
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
    show_results(orframe, outframe)
    return outframe

def process_video(input_path,original, output_path):
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

        #histogram_matching(frame[:, :, 0], frameOrig[:, :, 0])
        #histogram_matching(frame[:, :, 1], frameOrig[:, :, 1])
        #histogram_matching(frame[:, :, 2], frameOrig[:, :, 2])

        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.subtract(v, 30)
        #v = cv2.normalize(v,None,0,255,norm_type=cv2.NORM_MINMAX)
        #s = cv2.normalize(s,None,0,255,norm_type=cv2.NORM_MINMAX)
        s = cv2.add(s,40)
        v = np.clip(v, 0, 255)
        s = np.clip(s, 0, 255)
        final_hsv = cv2.merge((h, s, v))
        frame = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

        #output_0 = cv2.normalize(frame[:, :, 0],None,0,255,norm_type=cv2.NORM_MINMAX)
        #output_1 = cv2.normalize(frame[:, :, 1], None, 0, 255, norm_type=cv2.NORM_MINMAX)
        #output_2 = cv2.normalize(frame[:, :, 2], None, 0, 255, norm_type=cv2.NORM_MINMAX)
        #output_0 = np.clip(output_0, 0, 255)
        #output_1 = np.clip(output_1, 0, 255)
        #output_2 = np.clip(output_2, 0, 255)

        # Merge the color channels back together
        #frame = cv2.merge((frame[:, :, 0], frame[:, :, 1], frame[:, :, 2])).astype(np.uint8)

        # Write the processed frame
        out.write(frame)

        # Optional: Display the result (comment out for faster processing)
        cv2.imshow('Processing Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
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

    process_video("../DegradedVideos/archive_2017-01-07_President_Obama's_Weekly_Address.mp4","../SourceVideos/2017-01-07_President_Obama's_Weekly_Address.mp4", "output/output.mp4")
    #process_video("../DegradedVideos/archive_2017-01-07_President_Obama's_Weekly_Address.mp4", "output/output1.mp4")
    #process_video("../DegradedVideos/archive_2017-01-07_President_Obama's_Weekly_Address.mp4", "output/output2.mp4")

if __name__ == '__main__':
    main()