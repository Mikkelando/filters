import cv2
import numpy as np
import mediapipe as mp

def get_face_landmarks(frame, face_mesh):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    if results.multi_face_landmarks:
        return results.multi_face_landmarks[0].landmark
    return None

def get_landmark_points(landmarks, image_shape):
    ih, iw, _ = image_shape
    return [(int(landmark.x * iw), int(landmark.y * ih)) for landmark in landmarks]

def create_face_mask(image, landmarks):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    points = np.array(landmarks, dtype=np.int32)
    cv2.fillConvexPoly(mask, points, 255)
    return mask

def apply_histogram_matching(source, target, mask):
    matched = source.copy()

    # Ensure mask is of type np.uint8
    mask = mask.astype(np.uint8)

    for i in range(3):
        src_hist, bins = np.histogram(source[:, :, i][mask == 255], bins=256, range=(0, 256))
        tgt_hist, bins = np.histogram(target[:, :, i][mask == 255], bins=256, range=(0, 256))

        src_cdf = src_hist.cumsum()
        tgt_cdf = tgt_hist.cumsum()

        src_cdf_norm = src_cdf * (255.0 / src_cdf[-1])
        tgt_cdf_norm = tgt_cdf * (255.0 / tgt_cdf[-1])

        src_cdf_norm[src_cdf == 0] = 0
        tgt_cdf_norm[tgt_cdf == 0] = 0

        inverse_tgt_cdf = np.interp(tgt_cdf_norm, src_cdf_norm, np.arange(256))

        # Apply the LUT using the bitwise_and operation
        matched[:, :, i] = cv2.LUT(source[:, :, i], inverse_tgt_cdf.astype(np.uint8))
        matched[:, :, i] = cv2.bitwise_and(matched[:, :, i], mask)

    return matched




cap1 = cv2.VideoCapture('data/karin_out_00078_t0.2_final.mp4')
cap2 = cv2.VideoCapture('data/adj_Karin.mp4')

_, source_image = cap1.read()
_, target_image = cap2.read()

cap1.release()
cap2.release()

# src_shape = source_image.shape[:2]
# source_image = cv2.resize(source_image, target_image.shape[:2])

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

# Get face landmarks and create masks
source_landmarks = get_face_landmarks(source_image, face_mesh)
target_landmarks = get_face_landmarks(target_image, face_mesh)

if source_landmarks and target_landmarks:
    source_points = get_landmark_points(source_landmarks, source_image.shape)
    target_points = get_landmark_points(target_landmarks, target_image.shape)
    
    source_mask = create_face_mask(source_image, source_points)
    target_mask = create_face_mask(target_image, target_points)

    # Resize the masks to match the images if necessary
    if source_mask.shape != source_image.shape[:2]:
        source_mask = cv2.resize(source_mask, (source_image.shape[1], source_image.shape[0]))

    if target_mask.shape != target_image.shape[:2]:
        target_mask = cv2.resize(target_mask, (target_image.shape[1], target_image.shape[0]))

    source_image = cv2.resize(source_image, (1280, 720))
    # Apply histogram matching
    print('SHAPES: ', source_image.shape, target_image.shape, target_mask.shape)
    matched_image = apply_histogram_matching(source_image, target_image, target_mask)

    # Display the results
    cv2.imshow('Source Image', source_image)
    cv2.imshow('Target Image', target_image)
    cv2.imshow('Matched Image', matched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Face landmarks could not be detected in one or both images.")