import cv2
import numpy as np


def align_images(img1, img2):
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

def main():
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


if __name__ == '__main__':
    main()