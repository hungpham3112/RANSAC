import numpy as np
import cv2
from skimage.measure import ransac
from skimage.transform import SimilarityTransform

def ransac_alignment(keypoints1, keypoints2, matches, min_inliers=10, residual_threshold=5.0, max_trials=100):
    # Convert keypoints to (x, y) coordinates
    keypoints1_coords = np.array([keypoints1[m.queryIdx].pt for m in matches])
    keypoints2_coords = np.array([keypoints2[m.trainIdx].pt for m in matches])

    num_matches = len(matches)
    if num_matches < min_inliers:
        return None, []  # Return None and empty inlier mask

    # Apply RANSAC to estimate the similarity transform
    model, inliers = ransac((keypoints1_coords, keypoints2_coords), SimilarityTransform, min_samples=min_inliers,
                            residual_threshold=residual_threshold, max_trials=max_trials)

    if inliers is None or np.sum(inliers) < min_inliers:
        return None, []  # Return None and empty inlier mask

    # Return the estimated transformation matrix and the inlier mask
    return model.params, inliers

def extract_descriptors(image, keypoints):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Initialize SIFT detector and extractor
    sift = cv2.SIFT_create()

    # Compute keypoints and descriptors
    _, descriptors = sift.compute(gray, keypoints)

    return descriptors

def match_features(descriptors1, descriptors2):
    # Initialize FLANN matcher
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Perform matching
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Apply Lowe's ratio test to filter matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    return good_matches

def draw_matches(image1, image2, keypoints1, keypoints2, matches):
    # Convert images to BGR for OpenCV
    image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
    image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)

    # Create a blank image with enough space to display the two images side-by-side
    padding = 18
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2 + padding, 3), dtype=np.uint8)

    # Copy the original images into the visualization image
    vis[:h1, :w1, :3] = image1
    vis[:h2, w1 + padding:w1 + w2 + padding, :3] = image2

    # Draw the good matches with green keypoints
    for match in matches:
        color = (0, 255, 0)  # Green color
        pt1 = tuple(map(int, keypoints1[match.queryIdx].pt))
        pt2 = tuple(map(int, keypoints2[match.trainIdx].pt))
        pt2 = (pt2[0] + w1 + padding, pt2[1])  # Adjust x coordinate for image2
        cv2.circle(vis, pt1, 3, color)
        cv2.circle(vis, pt2, 3, color)

    # Convert back to RGB for streamlit
    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    return vis

def sift_feature_detector(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints
    keypoints = sift.detect(gray, None)

    # For drawing
    output_image = image.copy()
    # Draw keypoints on the image
    cv2.drawKeypoints(image, keypoints, output_image, color=(0, 255, 0))

    return keypoints, output_image
