import cv2
import numpy as np
import mediapipe as mp


def create_face_mask_with_contour(image, landmarks):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Extract x, y coordinates of all landmarks
    landmark_points = [(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])) for landmark in landmarks]

    # Find convex hull of the face landmarks (finds the boundary points in order)
    convex_hull_points = cv2.convexHull(np.array(landmark_points, dtype=np.int32))

    # Create a polygon from the convex hull points
    cv2.fillConvexPoly(mask, convex_hull_points, 255)

    return mask



def get_face_landmarks(image, face_mesh):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    if results.multi_face_landmarks:
        return results.multi_face_landmarks[0].landmark
    else:
        return None
if __name__ == "__main__":
    # Example image and landmarks (you need to have these set up)
    image = cv2.imread('matched_imgs/match_0.0.jpg')

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

    mediapipe_landmarks = get_face_landmarks(image, face_mesh)

    # Convert landmarks to mediapipe format (for example purposes)
    # class Landmark:
    #     def __init__(self, x, y):
    #         self.x = x
    #         self.y = y

    # mediapipe_landmarks = [Landmark(x, y) for x, y in landmarks]

    # Create face mask with contour
    face_mask = create_face_mask_with_contour(image, mediapipe_landmarks)

    # Display the mask (for visualization purposes)
    cv2.imshow('Face Mask', face_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
