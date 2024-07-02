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



if __name__ == "__main__":
        
    frames = get_video('data/karin_out_00078_t0.2_final.mp4')
    driving = np.array(get_video('data/Karin.mp4'))
    print('get videos')

    import mediapipe as mp

    print('crating face_detector')
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils


    faces = []
    bboxes = []
    print('detecting & extracting')
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
       output_size = frames[0].shape[:2]
       print(output_size)
       for frame in driving:
            
          
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                    
                    bboxes.append([ x, y, w, h])
                    face = frame[y:y+h, x:x+w]
                    face_resized = cv2.resize(face, output_size)
                    faces.append(face_resized)
                    break

     
 
    faces = np.array(faces)



    gen_faces = []
    gen_bboxes = []
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        output_size = frames[0].shape[:2]
        print(output_size)
        for frame in frames:
            
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                    # if x < 0 or y < 0 or x + w > iw or y + h > ih:
                    #     print("ALERT "*20)
                    #     break  # Skip invalid bounding boxes


                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, iw - x)
                    h = min(h, ih - y)

                    if w <= 0 or h <= 0:
                        print("ALERT "*5)
                        break  # Skip invalid bounding boxes

                   
                    
                    if face.size == 0:
                        print("ALERT "*5)
                        break  # Skip if the face region is empty

                    gen_bboxes.append([ x, y, w, h])
                    face = frame[y:y+h, x:x+w]
                    face_resized = cv2.resize(face, (512, 512))
                    gen_faces.append(face_resized)
                    break
    
    gen_faces = np.array(gen_faces)

    

    print('creating matched images')
    print(len(gen_faces), len(faces), len(frames))
    
    new_images = []
    for i in range(len(gen_faces)):

        # new_image = histogram_matching(frames[i][:, :, ::-1], driving[i][:, :, ::-1])
        new_image = match_histograms_moded(gen_faces[i][:, :, ::-1], faces[i][:, :, ::-1], strength=1.0)


        new_images.append(new_image.astype(np.uint8)[:, :, ::-1])

    print('get histes')

    cv2.imwrite('drive_face_0.jpg', faces[0])

    new_images = np.array(new_images)

    videowriter = cv2.VideoWriter('data/matched_gen.mp4', cv2.VideoWriter_fourcc(*'XVID'), 25, new_images.shape[1:3])
    for frame in new_images:
        videowriter.write(np.array(frame))
    videowriter.release()

    print('create video of gen')
    print('new_images[0].shape', new_images[0].shape)
    new_frames = []

    for drive, bbox, new_image in zip(driving, bboxes, new_images):
        new_frame = drive.copy()
        x, y, w, h = bbox
        half_h = h // 2

        new_image_resized = cv2.resize(new_image, (w, h))
        new_frame[y+half_h: y + h, x:x+w] = new_image_resized[half_h:, :, :]
        new_frames.append(new_frame)
    

    print('new_frames[0].shape', new_frames[0].shape)
    print('driving.shape[1:3]', driving.shape[1:3])
    videowriter1 = cv2.VideoWriter('data/new_final.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, driving.shape[1:3])
    for new_frame in new_frames:
        data = np.array(new_frame).copy()
        if data.shape != (720, 1280, 3):
            print('err: ',data.shape)
        videowriter1.write(data)
    videowriter1.release()

    print('create video of final')

    from tqdm import tqdm
    for i, new_frame in tqdm(enumerate(new_frames)):
        cv2.imwrite(fr'tmp_files\f_{i}.jpg', np.array(new_frame))