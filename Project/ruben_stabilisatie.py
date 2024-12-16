import time

import cv2
import numpy as np


def align_images_origineel(img1, img2):
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
    good_matches = matches[:10]

    # Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Estimate translation matrix
    translation_matrix, _ = cv2.estimateAffinePartial2D(dst_pts, src_pts)

    # Apply translation to the color image
    aligned_img2 = cv2.warpAffine(img2, translation_matrix, (img1.shape[1], img1.shape[0]))

    return aligned_img2, translation_matrix



def stabilize_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    _, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    transforms = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1, 3), np.float32)

    for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1):
        success, curr_frame = cap.read()
        if not success:
            break
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        height, width = curr_gray.shape
        y, x = np.mgrid[0:height:10, 0:width:10].reshape(2, -1)
        flow_points = np.column_stack((y, x))
        flow_vectors = flow[y, x]

        flow_points_prev = flow_points - flow_vectors
        flow_points_prev = np.clip(flow_points_prev, 0, [height - 1, width - 1])

        H, _ = cv2.findHomography(flow_points_prev, flow_points)

        curr_stabilized = cv2.warpPerspective(curr_frame, H, (width, height))
        out.write(curr_stabilized)

        prev_gray = curr_gray

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def main():
    timestamp = time.strftime("%d-%m-%Y_%H%M%S")

    # Voorbeeld gebruik
    input_video = "../DegradedVideos/archive_2017-01-07_President_Obama's_Weekly_Address.mp4"
    output_video = f"output/2017-01-07_President_Obama's_Weekly_Address_{timestamp}.mp4"
    stabilize_video(input_video, output_video)

"""
    # Read the images
    img1 = cv2.imread('output/classroom1.jpg')
    img2 = cv2.imread('output/classroom2.jpg')

    # Align images
    aligned_img2, translation_matrix = align_images(img1, img2)

    # Display results
    cv2.imshow('Original Image 1', img1)
    cv2.imshow('Original Image 2', img2)
    cv2.imshow('Aligned Image 2', aligned_img2)

    # Print translation matrix
    print("Translation Matrix:")
    print(translation_matrix)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
"""

if __name__ == '__main__':
    main()