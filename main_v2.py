import cv2
import numpy as np
import mediapipe as mp
from filter_utils import match_histograms_moded
from moviepy.editor import VideoFileClip

def get_video(path):
    frames = []
    cap = cv2.VideoCapture(path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS of the video '{path}': {fps}")
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame)

    cap.release()

    return frames

def change_fps(input_path, output_path, target_fps, target_frame_count):
    clip = VideoFileClip(input_path)
    new_clip = clip.set_fps(target_fps)
    target_duration = target_frame_count / target_fps
    if new_clip.duration > target_duration:
        new_clip = new_clip.subclip(0, target_duration)
    new_clip.write_videofile(output_path, fps=target_fps)

def get_face_landmarks(frame, face_mesh):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    if results.multi_face_landmarks:
        return results.multi_face_landmarks[0].landmark
    return None

if __name__ == "__main__":


    frames = get_video('data/karin_out_00078_t0.2_final.mp4')
    driving = np.array(get_video('data/adj_Karin.mp4'))
    print(len(driving), len(frames))
    print('get videos')

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

    faces = []
    face_landmarks_list = []

    for frame in driving:
        face_landmarks = get_face_landmarks(frame, face_mesh)
        if face_landmarks:
            ih, iw, _ = frame.shape
            face_landmarks_list.append([(int(landmark.x * iw), int(landmark.y * ih)) for landmark in face_landmarks])
            bbox_x = min([p[0] for p in face_landmarks_list[-1]])
            bbox_y = min([p[1] for p in face_landmarks_list[-1]])
            bbox_w = max([p[0] for p in face_landmarks_list[-1]]) - bbox_x
            bbox_h = max([p[1] for p in face_landmarks_list[-1]]) - bbox_y
            face = frame[bbox_y:bbox_y+bbox_h, bbox_x:bbox_x+bbox_w]
            face_resized = cv2.resize(face, (512, 512))
            faces.append(face_resized)

    gen_faces = []
    gen_face_landmarks_list = []

    for frame in frames:
        face_landmarks = get_face_landmarks(frame, face_mesh)
        if face_landmarks:
            ih, iw, _ = frame.shape
            gen_face_landmarks_list.append([(int(landmark.x * iw), int(landmark.y * ih)) for landmark in face_landmarks])
            bbox_x = min([p[0] for p in gen_face_landmarks_list[-1]])
            bbox_y = min([p[1] for p in gen_face_landmarks_list[-1]])
            bbox_w = max([p[0] for p in gen_face_landmarks_list[-1]]) - bbox_x
            bbox_h = max([p[1] for p in gen_face_landmarks_list[-1]]) - bbox_y
            face = frame[bbox_y:bbox_y+bbox_h, bbox_x:bbox_x+bbox_w]
            face_resized = cv2.resize(face, (512, 512))
            gen_faces.append(face_resized)

    print('creating matched images')
    new_images = []

    for i in range(len(faces)):
        new_image = match_histograms_moded(gen_faces[i][:, :, ::-1], faces[i][:, :, ::-1], strength=0.5)
        new_images.append(new_image.astype(np.uint8)[:, :, ::-1])

    print('get histes')
    new_images = np.array(new_images)
    videowriter = cv2.VideoWriter('data/matched_gen.mp4', cv2.VideoWriter_fourcc(*'XVID'), 25, new_images.shape[1:3])

    for frame in new_images:
        videowriter.write(np.array(frame))

    videowriter.release()
    print('create video of gen')

    new_frames = []

    for drive, orig_landmarks, gen_landmarks, new_image in zip(driving, face_landmarks_list, gen_face_landmarks_list, new_images):
        new_frame = drive.copy()
        new_image_resized = cv2.resize(new_image, (drive.shape[1], drive.shape[0]))

        for (orig_x, orig_y), (gen_x, gen_y) in zip(orig_landmarks, gen_landmarks):
            new_frame[orig_y:, orig_x] = new_image_resized[gen_y:, gen_x]

        new_frames.append(new_frame)

    print('new_frames[0].shape', new_frames[0].shape)
    print('driving.shape[1:3]', driving.shape[1:3])

    videowriter1 = cv2.VideoWriter('data/new_final.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, driving.shape[1:3])

    for new_frame in new_frames:
        data = np.array(new_frame).copy()
        if data.shape != (720, 1280, 3):
            print('err: ', data.shape)
        videowriter1.write(data)

    videowriter1.release()
    print('create video of final')

    from tqdm import tqdm
    for i, new_frame in tqdm(enumerate(new_frames)):
        cv2.imwrite(f'tmp_files/f_{i}.jpg', np.array(new_frame))
