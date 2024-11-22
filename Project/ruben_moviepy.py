from moviepy import VideoFileClip, concatenate_videoclips
import numpy as np
from utils import printProgressBar


def process_video(input_path, output_path):
    # Read the video clip
    clip = VideoFileClip(input_path)
    output_clip = VideoFileClip(output_path)

    # Function to process frames
    def process_frame(frame):
        # If this is the first frame in the sequence, initialize frames
        if not hasattr(process_frame, 'frames'):
            process_frame.frames = [frame] * 4
        else:
            # Shift frames and add new frame
            process_frame.frames = process_frame.frames[1:] + [frame]

        # Calculate median of last 4 frames
        stacked_frames = np.stack(process_frame.frames, axis=-1)
        return np.median(stacked_frames, axis=-1).astype(np.uint8)

    # Create a new clip from processed frames
    processed_clips = []
    max_iters = len(clip.size)
    for i,frame in enumerate(clip.iter_frames()):
        frame_processed = process_frame(frame)
        processed_clips.append(frame_processed)
        printProgressBar(i,max_iters)

    # Write the processed video
    fps = clip.fps
    clips = [VideoFileClip(input_path).duration(1 / fps).set_fps(fps).fl_image(lambda img: processed_clips.pop(0)) for _ in processed_clips]
    output_clip = concatenate_videoclips(clips)
    output_clip.write_videofile(output_path, codec='libx264')
    #output_clip.write_videofile(output_path, codec='libx264')
    """with clip.write_videofile(output_path, codec='libx264',
                              logger=None) as writer:
        for i, frame in enumerate(clip.iter_frames()):
            processed_frame = process_frame(frame)
            writer.write_frame(processed_frame)
            printProgressBar(i,max_iters)"""

    # Close clip to free up resources
    clip.close()

    # Close clip to free up resources
    clip.close()


def main():
    print("Processing video")
    process_video("../DegradedVideos/archive_2017-01-07_President_Obama's_Weekly_Address.mp4", "output/output.mp4")
    print("Klaar")


if __name__ == '__main__':
    main()