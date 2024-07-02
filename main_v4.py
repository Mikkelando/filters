import cv2
import numpy as np
import mediapipe as mp

from filter_utils import apply_histogram_matching

def get_face_landmarks(image, face_mesh):
    # Преобразуем изображение в RGB, так как Mediapipe использует RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Получаем результаты обнаружения лиц
    results = face_mesh.process(image_rgb)
    
    # Возвращаем ключевые точки лица (если они есть)
    if results.multi_face_landmarks:
        return results.multi_face_landmarks[0].landmark
    else:
        return None

def create_face_mask(image, landmarks):
    # Создаем черную маску с размерами изображения
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Получаем координаты ключевых точек лица
    points = np.array([(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])) for landmark in landmarks], dtype=np.int32)
    
    # Заполняем область лица на маске белым цветом
    cv2.fillConvexPoly(mask, points, 255)
    
    return mask

# def apply_histogram_matching(source_image, target_image, source_mask, target_mask, match_strength=0.5):
#     # Создаем копию исходного изображения
#     matched_image = source_image.copy()
    
#     # Применяем гистограммное соответствие для каждого канала
#     for i in range(3):
#         # Вычисляем гистограммы для областей, где маска равна 255 (белый цвет)
#         src_hist, _ = np.histogram(source_image[:, :, i][source_mask == 255], bins=256, range=(0, 256))
#         tgt_hist, _ = np.histogram(target_image[:, :, i][target_mask == 255], bins=256, range=(0, 256))
        
#         # Вычисляем кумулятивные распределения функций
#         src_cdf = src_hist.cumsum()
#         tgt_cdf = tgt_hist.cumsum()
        
#         # Нормализуем распределения
#         src_cdf = (src_cdf / src_cdf[-1]) * 255
#         tgt_cdf = (tgt_cdf / tgt_cdf[-1]) * 255
        
#         # Применяем гистограммное соответствие с использованием LUT
#         lut = np.interp(src_cdf, tgt_cdf, range(256))
        
#         # Управляем силой метчинга с помощью match_strength
#         # lut = source_image[:, :, i] * (1 - match_strength) + lut.astype(np.uint8) * match_strength
        
#         matched_image[:, :, i] = cv2.LUT(source_image[:, :, i], lut.astype(np.uint8)) *\
#             match_strength + (1 - match_strength) * source_image[:, :, i]
    
#     return matched_image


def apply_histogram_matching(gen_img, drive_img, match_strength=0.5):



    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

    # Получаем ключевые точки лиц на обоих изображениях
    gen_landmarks = get_face_landmarks(gen_img, face_mesh)
    drive_landmarks = get_face_landmarks(drive_img, face_mesh)


    # Создаем маски лиц для обоих изображений
    gen_mask = create_face_mask(gen_img, gen_landmarks)
    drive_mask = create_face_mask(drive_img, drive_landmarks)


    # Создаем копию исходного изображения
    matched_image = gen_img.copy()
    
    # Применяем гистограммное соответствие для каждого канала
    for i in range(3):
        # Вычисляем гистограммы для областей, где маска равна 255 (белый цвет)
        src_hist, _ = np.histogram(gen_img[:, :, i][gen_mask == 255], bins=256, range=(0, 256))
        tgt_hist, _ = np.histogram(gen_img[:, :, i][drive_mask == 255], bins=256, range=(0, 256))
        
        # Вычисляем кумулятивные распределения функций
        src_cdf = src_hist.cumsum()
        tgt_cdf = tgt_hist.cumsum()
        
        # Нормализуем распределения
        src_cdf = (src_cdf / src_cdf[-1]) * 255
        tgt_cdf = (tgt_cdf / tgt_cdf[-1]) * 255
        
        # Применяем гистограммное соответствие с использованием LUT
        lut = np.interp(src_cdf, tgt_cdf, range(256))
        
        # Управляем силой метчинга с помощью match_strength
        # lut = source_image[:, :, i] * (1 - match_strength) + lut.astype(np.uint8) * match_strength
        
        matched_image[:, :, i] = cv2.LUT(gen_img[:, :, i], lut.astype(np.uint8)) *\
            match_strength + (1 - match_strength) * gen_img[:, :, i]
    
    return matched_image




if __name__ == "__main__":
        
    # Загружаем изображения
    # gen_img = cv2.imread('gen_img.jpg')
    # drive_img = cv2.imread('drive_img.jpg')

    cap1 = cv2.VideoCapture('data/karin_out_00078_t0.2_final.mp4')
    cap2 = cv2.VideoCapture('data/adj_Karin.mp4')

    _, gen_img = cap1.read()
    _, drive_img = cap2.read()

    cap1.release()
    cap2.release()

    # Инициализируем Mediapipe FaceMesh
    # mp_face_mesh = mp.solutions.face_mesh
    # face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

    # # Получаем ключевые точки лиц на обоих изображениях
    # gen_landmarks = get_face_landmarks(gen_img, face_mesh)
    # drive_landmarks = get_face_landmarks(drive_img, face_mesh)

    # # Проверяем, что ключевые точки были найдены на обоих изображениях

    # # Создаем маски лиц для обоих изображений
    # gen_mask = create_face_mask(gen_img, gen_landmarks)
    # drive_mask = create_face_mask(drive_img, drive_landmarks)

    # Применяем гистограммное соответствие только для областей лиц
    match_strength = 0.7  # Пример силы метчинга (от 0 до 1)
    matched_img = apply_histogram_matching(gen_img, drive_img, match_strength)

    # Выводим маски и результаты
    # cv2.imshow('Generated Image', gen_img)
    # cv2.imshow('Driving Image', drive_img)
    # cv2.imshow('Generated Mask', gen_mask)
    # cv2.imshow('Driving Mask', drive_mask)
    cv2.imshow('Matched Image', matched_img)
    cv2.imwrite(f'matched_imgs/match_{match_strength}.jpg', matched_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

