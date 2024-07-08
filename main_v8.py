
import random
import matplotlib.pyplot as plt
import numpy as np
import csv
import cv2
from tqdm import tqdm
from experimental_filter import OneEuroFilter as OneEuroFilter_exp
from OneEuroFilter import OneEuroFilter

config = {
        'freq': 120,       # Hz
        'mincutoff': 1.0,  # Hz
        'beta': 0.1,
        'dcutoff': 1.0,
        'threshold' : 60
    }


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


def draw_landmarks_on_frames(frames, landmarks_list):
    frames_with_landmarks = []
    for frame, landmarks in zip(frames, landmarks_list):
        frame_copy = frame.copy()
        for (x, y) in landmarks:
            cv2.circle(frame_copy, (int(x), int(y)), 2, (0, 255, 0), -1)
        frames_with_landmarks.append(frame_copy)
    return frames_with_landmarks


def save_frames_to_video(frames, output_path, fps=25):
    height, width, layers = frames[0].shape
    size = (width, height)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for frame in frames:
        out.write(frame)
    out.release()
    print(f'saved . . . {output_path}  OK')

def load_landmarks(csv_path, qnt_l=468):

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        data_all = [row for row in reader]
    x_list = []
    y_list = []
    for row_index,row in enumerate(data_all[1:]):
        # frame_num = float(row[0])
        # if int(frame_num)!= row_index+1:
        #     return None
        x_list.append([float(x) for x in row[0:0+qnt_l]])
        y_list.append([float(y) for y in row[0+qnt_l:0+qnt_l + qnt_l]])
    x_array = np.array(x_list)
    y_array = np.array(y_list)
    landmark_array = np.stack([x_array,y_array],2)
    return landmark_array


def create_smooth(xs, n, filters=None):
    if filters is None:
        filters = [OneEuroFilter_exp(**config) for _ in range(n)]
    
    fn = [xs]
    for _ in range(n):
        f = []
        for i, x in enumerate(fn[-1]):
            f.append(filters[_](x, i)) 
        fn.append(f)
        # filter.reset()
    return fn[-1]

LND = load_landmarks('data/lnd.csv')
NEW_LND = []

filters = np.array([[(OneEuroFilter_exp(**config), OneEuroFilter_exp(**config)) for i in range(12)] for _ in range(len(LND[0]))])
XS = [LND[:, i, 0] for i in range(len(LND[0]))]
YS = [LND[:, i, 1] for i in range(len(LND[0]))]
tmp = []
for i, (xs, ys) in tqdm(enumerate(zip(XS, YS))):
    smoo_xs = create_smooth(xs, 12, filters=filters[i, :, 0])
    smoo_ys = create_smooth(ys, 12, filters=filters[i, :, 1])
    tmp.append([smoo_xs, smoo_ys])
tmp = np.array(tmp)
print(tmp.shape)
NEW_LND = np.transpose(tmp, (2, 0, 1))
print(NEW_LND.shape)
    




gen_imgs = get_video('data/karin_out_00078_t0.2_final.mp4')


frames_with_landmarks_1 = draw_landmarks_on_frames(gen_imgs, LND)
frames_with_landmarks_2 = draw_landmarks_on_frames(gen_imgs, NEW_LND)
save_frames_to_video(frames_with_landmarks_1, 'data/orig.mp4')
save_frames_to_video(frames_with_landmarks_2, f'data/smooth_tr{config["threshold"]}.mp4')
# save_frames_to_video(frames_with_landmarks_2, f'data/smooth_p12.mp4')