import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

from filter_utils import apply_histogram_matching_pre_exist, get_video

def get_face_landmarks(image, face_mesh):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    if results.multi_face_landmarks:
        return results.multi_face_landmarks[0].landmark
    else:
        return None



if __name__ == "__main__":
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

    
    gen_imgs = get_video('data/karin_out_00078_t0.2_final.mp4')
    drive_imgs = get_video('data/Karin.mp4')

    


    matched_imgs = []

    match_strength = 0.6 # Example match strength (0 to 1)
    for gen_img, drive_img in tqdm(zip(gen_imgs, drive_imgs)):

        gen_landmarks = get_face_landmarks(gen_img, face_mesh)
        drive_landmarks = get_face_landmarks(drive_img, face_mesh)


        matched_img = apply_histogram_matching_pre_exist(gen_img, drive_img, gen_landmarks, drive_landmarks, match_strength)

        if matched_img is not None:
            matched_imgs.append(matched_img)

    matched_imgs = np.array(matched_imgs)



    # if matched_img is not None:
    #     # cv2.imshow('Generated Image', gen_img)
    #     # cv2.imshow('Driving Image', drive_img)
    #     cv2.imshow('Matched Image', matched_img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    videowriter = cv2.VideoWriter(f'data/main_v6_Karin_matched_{match_strength}.mp4', cv2.VideoWriter_fourcc(*'XVID'), 25, matched_imgs.shape[1:3])

    for matched_img in tqdm(matched_imgs):
        videowriter.write(matched_img)

    videowriter.release()