import mediapipe as mp
import numpy as np
import cv2

rightEyeUpper1 = [247, 30, 29, 27, 28, 56, 190]
rightEyeLower1 = [130, 25, 110, 24, 23, 22, 26, 112, 243]
rightEyeUpper2 = [113, 225, 224, 223, 222, 221, 189]
rightEyeLower2 = [226, 31, 228, 229, 230, 231, 232, 233, 244]
rightEyeLower3 = [143, 111, 117, 118, 119, 120, 121, 128, 245]
rightEyebrowUpper = [156, 70, 63, 105, 66, 107, 55, 193]
rightEyebrowLower = [35, 124, 46, 53, 52, 65]
rightEyeIris = [473, 474, 475, 476, 477]
#leftEyeUpper0 = [466, 388, 387, 386, 385, 384, 398]
#leftEyeLower0 = [263, 249, 390, 373, 374, 380, 381, 382, 362]
leftEyeUpper1 = [467, 260, 259, 257, 258, 286, 414]
leftEyeLower1 = [359, 255, 339, 254, 253, 252, 256, 341, 463]
leftEyeUpper2 = [342, 445, 444, 443, 442, 441, 413]
leftEyeLower2 = [446, 261, 448, 449, 450, 451, 452, 453, 464]
leftEyeLower3 = [372, 340, 346, 347, 348, 349, 350, 357, 465]
#leftEyebrowUpper = [383, 300, 293, 334, 296, 336, 285, 417]
#leftEyebrowLower = [265, 353, 276, 283, 282, 295]
leftEyeIris = [468, 469, 470, 471, 472]
left_eye_mp = [*leftEyeUpper1,*leftEyeLower1,*leftEyeUpper2,*leftEyeLower2,*leftEyeLower3]
right_eye_mp = [*rightEyeUpper1,*rightEyeLower1,*rightEyeUpper2,*rightEyeLower2,*rightEyeLower3]
nose_mp = [68, 6, 197, 195, 5, 4, 1, 19, 94, 2,98, 97, 326, 327, 294, 278, 344, 440, 275, 45, 220, 115, 48, 64, 98]


def convert_face_landmark_to_5(face_landmark):
    if len(face_landmark) < 468:
        face_landmark_5 = np.array(
            [
                np.mean(np.matrix(face_landmark[36:42]), axis=0).tolist()[0],  # eye
                np.mean(np.matrix(face_landmark[42:48]), axis=0).tolist()[0],  # eye
                np.mean(np.matrix(face_landmark[27:36]), axis=0).tolist()[0],  # nose
            ])
    else:
        print(face_landmark[left_eye_mp].shape)
        face_landmark_5 = np.array(
            [
                np.mean(np.matrix(face_landmark[left_eye_mp]), axis=0).tolist()[0],  # eye
                np.mean(np.matrix(face_landmark[right_eye_mp]), axis=0).tolist()[0],  # eye
                np.mean(np.matrix(face_landmark[nose_mp]), axis=0).tolist()[0],  # nose
                # np.mean(np.matrix(face_landmark[brow_mp]), axis=0).tolist()[0],  # brow
            ])



    return face_landmark_5



def transformation_from_points(points1, points2):
    """
    Return an affine transformation [s * R | T] such that:

        sum ||s*R*p1,i + T - p2,i||^2

    is minimized.

    """
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T

    return np.vstack([np.hstack(((s2 / s1) * R,
                                        c2.T - (s2 / s1) * R * c1.T)),
                            np.matrix([0., 0., 1.])])

def warp_im(im, M, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)

    cv2.warpAffine(im,
                    M[:2],
                    (dshape[1], dshape[0]),
                    dst=output_im,
                    borderMode=cv2.BORDER_TRANSPARENT,
                    flags=cv2.WARP_INVERSE_MAP)
    return output_im


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

    source = cv2.imread('test_data/frame_from_model.jpg')[:, :, ::-1]
    target = cv2.imread('test_data/000047.jpg')

    
    s_landmarks =  np.array([(int(landmark.x * source.shape[1]), int(landmark.y * source.shape[0])) for landmark in get_face_landmarks(source, face_mesh)])
    t_landmarks = np.array([(int(landmark.x * target.shape[1]), int(landmark.y * target.shape[0])) for landmark in get_face_landmarks(target, face_mesh)])

    s_5_points = []
    t_5_points = []

    
    s_5_points.append(convert_face_landmark_to_5(s_landmarks))
    t_5_points.append(convert_face_landmark_to_5(t_landmarks))

    s_5_points = np.array(s_5_points)
    t_5_points = np.array(t_5_points)
    print(s_5_points.shape, t_5_points.shape )

    s_align =  np.matrix(s_5_points)
    t_align =  np.matrix(t_5_points)


    M = transformation_from_points(t_align, s_align)
    warped_im2 = warp_im(source, M, target.shape)

    cv2.imwrite('test_data/aff_matched.jpg', warped_im2)