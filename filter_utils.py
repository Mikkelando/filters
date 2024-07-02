import subprocess
import cv2
import os
import numpy as np
from skimage.exposure import match_histograms

from skimage._shared import utils 
import shutil
import mediapipe as mp



def blur_area(image, ksize=(7, 7), sigmaX=0):
    """
    Размывает заданную область на изображении.

    :param image: исходное изображение
    :param x: координата x верхнего левого угла области
    :param y: координата y верхнего левого угла области
    :param w: ширина области
    :param h: высота области
    :param ksize: размер ядра фильтра Гаусса (нечетное )
    :param sigmaX: стандартное отклонение по X для фильтра Гаусса
    :return: изображение с размытой областью
    """
    # Создаем копию изображения для размывания
    blurred_image = image.copy()

    # Применяем фильтр Гаусса к области
    blurred_image = cv2.GaussianBlur(blurred_image, ksize, sigmaX)

    
    return blurred_image



def histogram_matching(source, driving):
    
    
    return match_histograms(source, driving, channel_axis=-1)



def seamless_clone(original_image, generated_image, mask=None, center=None, method=cv2.NORMAL_CLONE):
    """
    Выполняет безшовное клонирование изображения.

    :param original_image_path: Путь к исходному изображению.
    :param generated_image_path: Путь к изображению для вставки.
    :param mask: Маска для вставки. Если None, будет создана маска, включающая все ненулевые пиксели изображения для вставки.
    :param center: Центр вставки. Если None, вставка будет выполнена в центр исходного изображения.
    :param method: Метод клонирования (cv2.NORMAL_CLONE или cv2.MIXED_CLONE).
    :return: Результирующее изображение с безшовной вставкой.
    """
    
    # Загрузка изображений
    # original_image = cv2.imread(original_image_path)
    # generated_image = cv2.imread(generated_image_path)

    # # Проверка на корректность загрузки изображений
    # if original_image is None or generated_image is None:
    #     raise ValueError("Невозможно загрузить одно или оба изображения. Проверьте пути к файлам.")

    # # Создание маски, если она не предоставлена
    # if mask is None:
    #     mask = np.zeros(generated_image.shape[:2], np.uint8)
    #     mask[generated_image[:, :, 0] > 0] = 255

    # Определение центра вставки, если он не указан
    # if center is None:
    #     center = (original_image.shape[1] // 2, original_image.shape[0] // 2)

    # Применение безшовного клонирования
    result = cv2.seamlessClone(generated_image, original_image, mask, center, method)

    return result









def _match_cumulative_cdf(source, template, strength=1.0):
    """
    Return modified source array so that the cumulative density function of
    its values matches the cumulative density function of the template.

    Parameters
    ----------
    source : ndarray
        Input source array.
    template : ndarray
        Template array for histogram matching.
    strength : float
        Strength of the histogram matching effect (0 to 1).
    """
    if source.dtype.kind == 'u':
        src_lookup = source.reshape(-1)
        src_counts = np.bincount(src_lookup)
        tmpl_counts = np.bincount(template.reshape(-1))

        # omit values where the count was 0
        tmpl_values = np.nonzero(tmpl_counts)[0]
        tmpl_counts = tmpl_counts[tmpl_values]
    else:
        src_values, src_lookup, src_counts = np.unique(
            source.reshape(-1), return_inverse=True, return_counts=True
        )
        tmpl_values, tmpl_counts = np.unique(template.reshape(-1), return_counts=True)

    # calculate normalized quantiles for each array
    src_quantiles = np.cumsum(src_counts) / source.size
    tmpl_quantiles = np.cumsum(tmpl_counts) / template.size

    interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
    
    # Adjust the source values with the calculated strength
    matched_values = (1 - strength) * source.reshape(-1) + strength * interp_a_values[src_lookup]
    return matched_values.reshape(source.shape)

@utils.channel_as_last_axis(channel_arg_positions=(0, 1))
def match_histograms_moded(image, reference, *, strength=1.0, channel_axis=-1):
    """Adjust an image so that its cumulative histogram matches that of another.

    The adjustment is applied separately for each channel.

    Parameters
    ----------
    image : ndarray
        Input image. Can be gray-scale or in color.
    reference : ndarray
        Image to match histogram of. Must have the same number of channels as
        image.
    strength : float, optional
        Strength of the histogram matching effect (0 to 1). Default is 1.0.
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

    Returns
    -------
    matched : ndarray
        Transformed input image.

    Raises
    ------
    ValueError
        Thrown when the number of channels in the input image and the reference
        differ.

    References
    ----------
    .. [1] http://paulbourke.net/miscellaneous/equalisation/

    """
    if image.ndim != reference.ndim:
        raise ValueError(
            'Image and reference must have the same number of channels.'
        )

    if channel_axis is not None:
        if image.shape[-1] != reference.shape[-1]:
            raise ValueError(
                'Number of channels in the input image and '
                'reference image must match!'
            )

        matched = np.empty(image.shape, dtype=image.dtype)
        for channel in range(image.shape[-1]):
            matched_channel = _match_cumulative_cdf(
                image[..., channel], reference[..., channel], strength=strength
            )
            matched[..., channel] = matched_channel
    else:
        # _match_cumulative_cdf will always return float64 due to np.interp
        matched = _match_cumulative_cdf(image, reference, strength=strength)

    if matched.dtype.kind == 'f':
        # output a float32 result when the input is float16 or float32
        out_dtype = utils._supported_float_type(image.dtype)
        matched = matched.astype(out_dtype, copy=False)
    return matched









def get_face_landmarks(image, face_mesh):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    if results.multi_face_landmarks:
        return results.multi_face_landmarks[0].landmark
    else:
        return None

def create_face_mask(image, landmarks):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    points = np.array([(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])) for landmark in landmarks], dtype=np.int32)
    cv2.fillConvexPoly(mask, points, 255)
    return mask

def apply_histogram_matching(gen_img, drive_img, match_strength=0.5):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
    gen_landmarks = get_face_landmarks(gen_img, face_mesh)
    drive_landmarks = get_face_landmarks(drive_img, face_mesh)


    if gen_landmarks and drive_landmarks:
        
        gen_mask = create_face_mask(gen_img, gen_landmarks)
        drive_mask = create_face_mask(drive_img, drive_landmarks)

        matched_image = gen_img.copy()

        for i in range(3):
            src_hist, _ = np.histogram(gen_img[:, :, i][gen_mask == 255], bins=256, range=(0, 256))
            tgt_hist, _ = np.histogram(drive_img[:, :, i][drive_mask == 255], bins=256, range=(0, 256))

            src_cdf = src_hist.cumsum()
            tgt_cdf = tgt_hist.cumsum()

            src_cdf = (src_cdf / src_cdf[-1]) * 255
            tgt_cdf = (tgt_cdf / tgt_cdf[-1]) * 255

            lut = np.interp(src_cdf, tgt_cdf, range(256))
        
           
            
            matched_image[:, :, i] = cv2.LUT(gen_img[:, :, i], lut.astype(np.uint8)) *\
                match_strength + (1 - match_strength) * gen_img[:, :, i]

        return matched_image

    else:
        print("Face landmarks not detected in one or both images.")
        return None
    



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




def apply_histogram_matching_pre_exist(gen_img, drive_img, gen_landmarks, drive_landmarks, match_strength=0.5):
    gen_mask = create_face_mask(gen_img, gen_landmarks)
    drive_mask = create_face_mask(drive_img, drive_landmarks)
    matched_image = gen_img.copy()
    for i in range(3):
        src_hist, _ = np.histogram(gen_img[:, :, i][gen_mask == 255], bins=256, range=(0, 256))
        tgt_hist, _ = np.histogram(drive_img[:, :, i][drive_mask == 255], bins=256, range=(0, 256))
        src_cdf = src_hist.cumsum()
        tgt_cdf = tgt_hist.cumsum()
        src_cdf = (src_cdf / src_cdf[-1]) * 255
        tgt_cdf = (tgt_cdf / tgt_cdf[-1]) * 255
        lut = np.interp(src_cdf, tgt_cdf, range(256))
        matched_image[:, :, i] = cv2.LUT(gen_img[:, :, i], lut.astype(np.uint8)) *\
                match_strength + (1 - match_strength) * gen_img[:, :, i]

        return matched_image
