import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

from filter_utils import apply_histogram_matching, get_video

# Example usage:
if __name__ == "__main__":
    # cap1 = cv2.VideoCapture('data/karin_out_00078_t0.2_final.mp4')
    # cap2 = cv2.VideoCapture('data/adj_Karin.mp4')


    
    # _, gen_img = cap1.read()
    # _, drive_img = cap2.read()

    # cap1.release()
    # cap2.release()

    gen_imgs = get_video('data/karin_out_00078_t0.2_final.mp4')
    drive_imgs = get_video('data/tmp_video.mp4')


    matched_imgs = []

    match_strength = 1.0  # Example match strength (0 to 1)
    for gen_img, drive_img in tqdm(zip(gen_imgs, drive_imgs)):

        matched_img = apply_histogram_matching(gen_img, drive_img, match_strength)
        if matched_img is not None:
            matched_imgs.append(matched_img)

    matched_imgs = np.array(matched_imgs)



    # if matched_img is not None:
    #     # cv2.imshow('Generated Image', gen_img)
    #     # cv2.imshow('Driving Image', drive_img)
    #     cv2.imshow('Matched Image', matched_img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    videowriter = cv2.VideoWriter(f'data/Karin_matched_{match_strength}.mp4', cv2.VideoWriter_fourcc(*'XVID'), 25, matched_imgs.shape[1:3])

    for matched_img in tqdm(matched_imgs):
        videowriter.write(matched_img)

    videowriter.release()
