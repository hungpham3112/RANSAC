import streamlit as st
from PIL import Image
import numpy as np
from .lib import sift_feature_detector, extract_descriptors, match_features, draw_matches, ransac_alignment

def main():
    st.title("Panorama for two images")
    uploaded_files = st.sidebar.file_uploader("", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    warp_col1, warp_col2 = st.columns([2, 1])
    if uploaded_files:
        if len(uploaded_files) >= 2:
            with warp_col1:
                col1, col2 = st.columns(2)
                with col1:
                    placeholder1 = st.empty()
                    placeholder1.header("Image 1")
                    placeholder1.image(uploaded_files[1], use_column_width=True)
                with col2:
                    placeholder2 = st.empty()
                    placeholder2.header("Image 2")
                    placeholder2.image(uploaded_files[0], use_column_width=True)
        else:
            st.write("Please upload at least two images.")
    else:
        st.empty()

    with warp_col2:
        if warp_col2.button("Feature detection"):
            placeholder1.empty()
            placeholder2.empty()

            image1 = np.array(Image.open(uploaded_files[1]))
            image2 = np.array(Image.open(uploaded_files[0]))

            keypoints1, image1_with_keypoints = sift_feature_detector(image1)
            keypoints2, image2_with_keypoints = sift_feature_detector(image2)

            placeholder1.image(image1_with_keypoints, use_column_width=True)
            placeholder2.image(image2_with_keypoints, use_column_width=True)

            st.session_state.keypoints1 = keypoints1
            st.session_state.keypoints2 = keypoints2
            st.session_state.image1 = image1
            st.session_state.image2 = image2

        if warp_col2.button("Remove outliers with RANSAC"):
            placeholder1.empty()
            placeholder2.empty()
            image1 = np.array(Image.open(uploaded_files[1]))
            image2 = np.array(Image.open(uploaded_files[0]))

            keypoints1 = st.session_state.keypoints1
            keypoints2 = st.session_state.keypoints2

            # Extract descriptors from keypoints
            descriptors1 = extract_descriptors(image1, keypoints1)
            descriptors2 = extract_descriptors(image2, keypoints2)

            # Match features
            matches = match_features(descriptors1, descriptors2)

            # Apply RANSAC to estimate transformation matrix
            transform, inliers = ransac_alignment(keypoints1, keypoints2, matches)

            if transform is not None:
                # Filter inlier matches
                filtered_matches = [m for m, inlier in zip(matches, inliers) if inlier]

                img_match = draw_matches(image1, image2, keypoints1, keypoints2, filtered_matches)
                warp_col1.image(img_match, use_column_width=True)
            else:
                st.write("Not enough inliers to estimate the alignment.")

        if warp_col2.button("Reset"):
            placeholder1.image(uploaded_files[1], use_column_width=True)
            placeholder2.image(uploaded_files[0], use_column_width=True)

if __name__ == "__main__":
    main()
