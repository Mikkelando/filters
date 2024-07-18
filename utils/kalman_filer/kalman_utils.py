import csv
from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
from pykalman import KalmanFilter

def calculate_gradients(XS, YS):
    """
    Рассчитывает производные для каждой точки по каждому кадру.

    Параметры:
    XS (numpy.array): Массив координат X для каждой точки по каждому кадру. Формат (M, n).
    YS (numpy.array): Массив координат Y для каждой точки по каждому кадру. Формат (M, n).

    Возвращает:
    numpy.array: Массив градиентов по X и Y для каждой точки. Формат (M, n-1, 2).
    """
    grad_X = np.diff(XS, axis=1)  # Производные по X
    grad_Y = np.diff(YS, axis=1)  # Производные по Y
    
    gradients = np.stack((grad_X, grad_Y), axis=-1)  # Объединяем градиенты в один массив (M, n-1, 2)
    
    return gradients

def adaptive_denoise(XS, YS, sigma_large=0.1, sigma_small=10, alpha_large=0.1, alpha_small=0.99, threshold=1):
    """
    Применяет адаптивное сглаживание к координатам точек.

    Параметры:
    XS (numpy.array): Массив координат X для каждой точки по каждому кадру. Формат (M, n).
    YS (numpy.array): Массив координат Y для каждой точки по каждому кадру. Формат (M, n).
    sigma_large (float): Сигма для гауссового фильтра при сильном движении.
    sigma_small (float): Сигма для гауссового фильтра при слабом движении.
    threshold (float): Порог для определения сильного и слабого движения.

    Возвращает:
    numpy.array, numpy.array: Отфильтрованные координаты X и Y. Формат (M, n).
    """
    gradients = calculate_gradients(XS, YS)
    gradient_magnitudes = np.linalg.norm(gradients, axis=-1)  # Модули градиентов (M, n-1)

    # Добавляем нулевой столбец в начало, чтобы размеры совпадали с оригинальными XS и YS
    gradient_magnitudes = np.hstack([np.zeros((gradients.shape[0], 1)), gradient_magnitudes])

    XS_denoised = np.copy(XS)
    YS_denoised = np.copy(YS)
    weak = np.zeros(XS.shape[:2], dtype=np.bool)
    
    # for i in range(XS.shape[0]):
    #     for j in range(XS.shape[1]):
    #         if gradient_magnitudes[i, j] > threshold:
    #             sigma = sigma_large
    #             weak[i,j] = False
    #         else:
    #             sigma = sigma_small
    #             weak[i,j] = True

    #         XS_denoised[i, j] = gaussian_filter1d(XS[i, :], sigma=sigma)[j]
    #         YS_denoised[i, j] = gaussian_filter1d(YS[i, :], sigma=sigma)[j]
    
    # return XS_denoised, YS_denoised, weak


    for i in range(XS.shape[0]):
        for j in range(1, XS.shape[1]):
            if gradient_magnitudes[i, j] > threshold:
                alpha = alpha_large
                weak[i,j] = False
            else:
                alpha = alpha_small
                weak[i,j] = True

            XS_denoised[i, j] = alpha * XS_denoised[i, j] + (1 - alpha) * XS_denoised[i, j - 1]
            YS_denoised[i, j] = alpha * YS_denoised[i, j] + (1 - alpha) * YS_denoised[i, j - 1]
    
    return XS_denoised, YS_denoised, weak

def adaptive_kalman_smoothing(XS, YS, R_large=1.0, R_small=0.1, threshold=0.5):
  
    gradients = calculate_gradients(XS, YS)
    gradient_magnitudes = np.linalg.norm(gradients, axis=-1)  # Модули градиентов (M, n-1)

    gradient_magnitudes = np.hstack([np.zeros((gradients.shape[0], 1)), gradient_magnitudes])

    XS_denoised = np.copy(XS)
    YS_denoised = np.copy(YS)
    
    for i in tqdm(range(XS.shape[0])):
     
        # transition_matrices=np.array([[1]]),
        # observation_matrices=np.ones((len(XS[0]),1)),
        # initial_state_mean=XS[i, 0],
        # initial_state_covariance=1.0,
        # transition_covariance=0.1,
        # # observation_covariance=R
        

        # R = np.where(gradient_magnitudes[i, :] > threshold, R_large, R_small)
        
        kf = KalmanFilter(

            initial_state_mean=XS[i, 0],  # Начальное состояние
            
        )
        
        # Применение фильтра Калмана к координатам X
        
        state_means, _ = kf.smooth(XS[i, :])
        XS_denoised[i, :] = state_means.flatten()
        
        # Обновление начальных состояний для Y
        initial_state_mean = YS[i, 0]
        kf = KalmanFilter(
           
            initial_state_mean= YS[i, 0],
           
        )
        
        # Применение фильтра Калмана к координатам Y
        state_means, _ = kf.smooth(YS[i, :])
        YS_denoised[i, :] = state_means.flatten()
    
    return XS_denoised, YS_denoised


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



def check_pos(xs, xs_nose):
    
    for i in range(len(xs)):
        if xs[i] > xs_nose[i]:
            return False
    return True




def klmn_filter(LND, POWER=2, indicies= None):

    if indicies is None:
        indicies = [i for i in range(len(LND[0]))]
    XS = np.array([LND[:, i, 0] for i in range(len(LND[0]))])[indicies]
    YS = np.array([LND[:, i, 1] for i in range(len(LND[0]))])[indicies]

    XS, YS = adaptive_kalman_smoothing(np.array(XS), np.array(YS))
    for _ in range(POWER - 1):
        XS, YS = adaptive_kalman_smoothing(np.array(XS), np.array(YS))

    data = np.array([np.array(XS), np.array(YS)])
    NEW_LND = np.transpose(data, (2,1,0))
    return NEW_LND


def stream_klmn_filter(lnd, filters, prev_value, prev_state_covariance=[[np.eye(1), np.eye(1)] for _ in range(468)] , power=4):
    curr_data = []
    
    for i, pt in enumerate(lnd):
        filter = filters[i]
        x = pt[0]
        y = pt[1]

        (x_smoo, smoothed_state_covariances_x) = filter[0].filter_update(prev_value[i][0], prev_state_covariance[i][0], x)
        (y_smoo, smoothed_state_covariances_y) = filter[0].filter_update(prev_value[i][1], prev_state_covariance[i][1], y)

        curr_data.append([[x_smoo, smoothed_state_covariances_x], [y_smoo, smoothed_state_covariances_y]])
    
    return curr_data


def get_filters(LND):
    return [[KalmanFilter(initial_state_mean=LND[0, i, 0]) ,KalmanFilter(initial_state_mean=LND[0, i, 1]) ]  for i in range(len(LND[0]))]


if __name__ == "__main__":
    LND = np.array(load_landmarks('data/joe_face_lnd.csv'))
    XS = np.array([LND[:, i, 0] for i in range(len(LND[0]))])
    YS = np.array([LND[:, i, 1] for i in range(len(LND[0]))])

    filters = get_filters(LND)
    NEW_LND = []
    NEW_LND.append(LND[0])
    for i, lnd in tqdm(enumerate(LND[1:])):
        if i == 0:
            data = stream_klmn_filter(lnd, filters, LND[0] )
            # print(i, np.array([ [data[j][0][0], data[j][1][0]] for j in range(len(LND[0])) ]).shape)
            # print([ [data[j][0][0], data[j][1][0]] for j in range(len(LND[0])) ])

            # print(data[0][0][0][0][0])
            NEW_LND.append(np.array([ [data[j][0][0][0][0], data[j][1][0][0][0]] for j in range(len(LND[0])) ]))
        else:
            data = stream_klmn_filter(lnd, filters, [ [data[i][0][0], data[i][1][0]] for i in range(len(LND[0])) ],
                                       prev_state_covariance= [ [data[i][0][1], data[i][1][1]] for i in range(len(LND[0])) ])
            
            print(i, np.array([ [data[j][0][0][0][0], data[j][1][0][0][0]] for j in range(len(LND[0])) ]).shape)
            NEW_LND.append(np.array([ [data[j][0][0][0][0], data[j][1][0][0][0]] for j in range(len(LND[0])) ]))
          

    NEW_LND = np.array(NEW_LND)
    XS_NEW =  np.array([NEW_LND[:, i, 0] for i in range(len(NEW_LND[0]))])

    print(NEW_LND.shape)
    print(XS_NEW.shape)
    

    plt.plot(XS[0])
    plt.plot(XS_NEW[0])
    plt.show()







    # XSa = []
    # YSa = []


    # for i, (xs, ys) in tqdm(enumerate(zip(XS, YS))):
    #     if check_pos(ys, YS[1]):

    #         XSa.append(xs)
    #         YSa.append(ys)
    #         # _ , large_jumps_x, threshold_x = split_into_clusters_v2(xs)
    #         # _ , large_jumps_y, threshold_y = split_into_clusters_v2(ys)
    #         # m_x = [1 if i in large_jumps_x else 0 for i in range(len(xs))] 
    #         # m_y = [1 if i in large_jumps_y else 0 for i in range(len(ys))] 
    
    # XSa, YSa = np.array(XSa), np.array(YSa)
    # # Пример использования:
    # # Замените XS и YS на ваши реальные данные
    # M = 468  # Количество точек
    # n = 10  # Количество кадров

    # XS = np.random.rand(M, n)  # Пример случайного массива координат X
    # YS = np.random.rand(M, n)  # Пример случайного массива координат Y

    # Применяем адаптивное сглаживание
    # XS_denoised, YS_denoised = adaptive_kalman_smoothing(XSa, YSa)


    # print(XS_denoised.shape)
    # print(YS_denoised.shape)

    # plt.plot(XS_denoised[0])
    # plt.plot(XS[0])
    # print(weak[0])
    # plt.scatter(np.arange(len(XS[0]))[weak[0]], XS[0][weak[0]])
    # plt.show()
    # # Вывод результатов для первых 3 точек
    # for point_idx in range(3):
    #     print(f"Точка {point_idx + 1}:")
    #     print(f"  Исходные координаты X: {XS[point_idx]}")
    #     print(f"  Отфильтрованные координаты X: {XS_denoised[point_idx]}")
    #     print(f"  Исходные координаты Y: {YS[point_idx]}")
    #     print(f"  Отфильтрованные координаты Y: {YS_denoised[point_idx]}")