from filter_utils import smooth_lnd_for_video, get_video
import mediapipe as mp
import cv2
from tqdm import tqdm

def get_face_landmarks(image, face_mesh):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    if results.multi_face_landmarks:
        return results.multi_face_landmarks[0].landmark
    else:
        return None


def draw_landmarks_on_frames(frames, landmarks_list):
    frames_with_landmarks = []
    for frame, landmarks in zip(frames, landmarks_list):
        frame_copy = frame.copy()
        for (x, y) in landmarks:
            cv2.circle(frame_copy, (x, y), 2, (0, 255, 0), -1)
        frames_with_landmarks.append(frame_copy)
    return frames_with_landmarks


def save_frames_to_video(frames, output_path, fps=25):
    height, width, layers = frames[0].shape
    size = (width, height)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for frame in frames:
        out.write(frame)
    out.release()




if __name__ == "__main__":
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

    
    gen_imgs = get_video('data/karin_out_00078_t0.2_final.mp4')


    LND = []

    
    for gen_img in tqdm(gen_imgs):

        gen_landmarks = get_face_landmarks(gen_img, face_mesh)
        

        gen_landmarks = [(int(landmark.x * gen_img.shape[1]), int(landmark.y * gen_img.shape[0])) for landmark in gen_landmarks]
        LND.append(gen_landmarks)

        
    print('PROCESS')
    NEW_LND = smooth_lnd_for_video(gen_imgs, LND, power =5, fps = 25)


    frames_with_landmarks_1 = draw_landmarks_on_frames(gen_imgs, LND)
    frames_with_landmarks_2 = draw_landmarks_on_frames(gen_imgs, NEW_LND)
    save_frames_to_video(frames_with_landmarks_1, 'data/orig.mp4')
    save_frames_to_video(frames_with_landmarks_2, 'data/smooth.mp4')
