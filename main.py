import cv2

from filter_utils import histogram_matching, match_histograms_moded
import numpy as np

def get_video(path):
    frames = []
    cap = cv2.VideoCapture(path)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame)

    cap.release()

    return frames

frames = get_video('data/karin_out_00078_t0.2_final.mp4')
driving = get_video('data/Karin.mp4')
print('get videos')


new_images = []
for i in range(len(frames)):

    # new_image = histogram_matching(frames[i][:, :, ::-1], driving[i][:, :, ::-1])
    new_image = match_histograms_moded(frames[i][:, :, ::-1], driving[i][:, :, ::-1], strength=0.5)


    new_images.append(new_image.astype(np.uint8)[:, :, ::-1])

print('get histes')



new_images = np.array(new_images)

videowriter = cv2.VideoWriter('data/new_Karin.mp4', cv2.VideoWriter_fourcc(*'XVID'), 25, new_images.shape[1:3])
for frame in new_images:
    videowriter.write(np.array(frame))
videowriter.release()

print('create video')

