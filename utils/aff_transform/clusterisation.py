
import csv
import cv2
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm


def cluster_landmarks(XS, YS,  num_clusters=6):
    """
    Кластеризует лендмарки на каждом кадре видео и возвращает метки кластеров для каждого кадра.
    
    Параметры:
    frames_landmarks (list): Список лендмарок для каждого кадра видео. Каждый элемент списка представляет собой список точек (x, y).
    num_clusters (int): Количество кластеров для кластеризации.

    Возвращает:
    list: Список меток кластеров для каждого кадра видео.
    """

    # Подготовка данных для кластеризации
    cluster_labels_all_frames = []

    for i in range(len(XS[0])):
        xs = XS[:, i]
        ys = YS[:, i]
        X = np.array ( [(xs[j], ys[j]) for j in range(len(xs))] )
        # X = np.array([ (x, y) for x,y in zip(XS[i], YS[i])] )
        # # Преобразование лендмарок в формат, пригодный для кластеризации
        # X = np.array(landmarks)

        # Кластеризация с помощью K-means
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(X)

        # Получение меток кластеров для текущего кадра
        cluster_labels = kmeans.labels_
        cluster_labels_all_frames.append(cluster_labels)

    return cluster_labels_all_frames




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


def lucas_kanade_1d(arr, window_size=5):
    """
    Применяет адаптированный алгоритм Лукса-Канаде к одномерному массиву для нахождения больших скачков.

    :param arr: Входной одномерный массив.
    :param window_size: Размер окна для расчета градиента.
    :return: Массив скоростей (разниц) для каждого элемента массива.
    """
    n = len(arr)
    half_window = window_size // 2

    # Градиенты
    gradients = np.zeros(n)

    for i in range(half_window, n - half_window):
        # Центральное окно
        window = arr[i - half_window:i + half_window + 1]

        # Градиент
        gradient = np.gradient(window)
        gradients[i] = np.mean(gradient)
    gradient[0] = gradient[1]
    gradient[-1] = gradient[-2]
    return gradients
def find_most_stable_points_per_cluster(G, cluster_labels, num_clusters=6):
    """
    Находит наиболее стабильные точки для каждого кластера на каждом кадре.

    Параметры:
    G (numpy.array): Градиенты для каждой точки на каждом кадре. Формат (n, 468, 2).
    cluster_labels (numpy.array): Массив с метками кластеров для каждой точки на каждом кадре. Формат (n, 468).
    num_clusters (int): Количество кластеров.

    Возвращает:
    numpy.array: Массив наиболее стабильных точек для каждого кластера на каждом кадре. Формат (n, 6).
    """
    n_frames, n_points, _ = G.shape
    stable_points_indices = np.zeros((n_frames, num_clusters), dtype=int)

    for frame_idx in range(n_frames):
        frame_gradients = G[frame_idx]  # Градиенты для текущего кадра
        frame_labels = cluster_labels[frame_idx]  # Метки кластеров для текущего кадра
        
        for cluster in range(num_clusters):
            # Выбираем индексы точек текущего кластера
            cluster_indices = np.where(frame_labels == cluster)[0]
            
            if len(cluster_indices) > 0:
                # Выбираем градиенты точек текущего кластера
                cluster_gradients = frame_gradients[cluster_indices]

                # Вычисляем суммы градиентов по обоим направлениям
                gradient_magnitudes = np.linalg.norm(cluster_gradients, axis=1)

                # Находим индекс точки с минимальным градиентом
                most_stable_point_index = cluster_indices[np.argmin(gradient_magnitudes)]
                
                # Записываем индекс наиболее стабильной точки
                stable_points_indices[frame_idx, cluster] = most_stable_point_index
            else:
                # Если в кластере нет точек, устанавливаем значение -1
                stable_points_indices[frame_idx, cluster] = -1

    return stable_points_indices



def find_best_gradients(gradients, labels, num_clusters=6):
    """
    Находит наиболее стабильные градиенты для каждого кластера.

    Параметры:
    gradients (numpy.array): Массив градиентов для каждой точки. Формат (468, 2).
    labels (numpy.array): Массив с метками кластеров для каждой точки. Формат (468,).
    num_clusters (int): Количество кластеров.

    Возвращает:
    numpy.array: Массив наилучших градиентов для каждого кластера. Формат (num_clusters, 2).
    """
    best_gradients = np.zeros(num_clusters)
    
    for cluster in range(num_clusters):
        # Выбираем градиенты точек текущего кластера
        cluster_indices = np.where(labels == cluster)[0]
        
        if len(cluster_indices) > 0:
            cluster_gradients = gradients[cluster_indices]
            
            # Вычисляем сумму модулей градиентов
            gradient_magnitudes = np.linalg.norm(cluster_gradients, axis=1)
            # gradient_magnitudes = np.linalg.norm(cluster_gradients)
            # Находим индекс точки с минимальным градиентом
            # print(len(gradient_magnitudes))
            best_index = np.argmin(gradient_magnitudes)
            best_gradients[cluster] = best_index
        else:
            # Если в кластере нет точек, устанавливаем градиент в (0, 0)
            best_gradients[cluster] = 0
    
    return best_gradients

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
    
    # gradients = np.stack(np.array([grad_X, grad_Y]), axis=-1)  # Объединяем градиенты в один массив (M, n-1, 2)
    gradients = np.array([grad_X, grad_Y])
    
    return gradients
def find_most_stable_points(XS, YS, clustered_frames):

    # XS = [LND[:, i, 0] for i in range(len(LND[0]))]
    # YS = [LND[:, i, 1] for i in range(len(LND[0]))]

    grad_X = [lucas_kanade_1d(xs) for xs in XS]
    grad_Y = [lucas_kanade_1d(ys) for ys in YS]

    gradients = calculate_gradients(XS, YS)
    print('gradients.shape: ', gradients.shape)

    expanded_gradients = np.concatenate((gradients[..., :1], gradients), axis=-1)
    transposed_gradients = np.transpose(expanded_gradients, (2, 1, 0))
    # gradients = np.transpose(gradients, (2, 1, 0))
    print('transposed_gradients.shape', transposed_gradients.shape)
    # gradient_magnitudes = np.linalg.norm(gradients, axis=-1)  # Модули градиентов (M, n-1)
    # gradient_magnitudes = np.hstack([np.zeros((gradients.shape[0], 1)), gradient_magnitudes])
    # print('gradient_magnitudes.shape: ', gradient_magnitudes.shape)
    G = np.array([[grad_x, grad_y] for (grad_x, grad_y) in zip(grad_X, grad_Y)]) # (468, 2, n)
    G = np.transpose(G, (2, 0, 1))
    print('G.shape', G.shape)
    # gradient_magnitudes = np.transpose(gradient_magnitudes, (1, 0))
    
    ans = []
    # print("G", G.shape)
    # print('gradient_magnitudes', gradient_magnitudes.shape)
    cluster_labels = clustered_frames[0]
    # print(cluster_labels.shape)
    for grads, frame_labels in zip(transposed_gradients, clustered_frames):
        # print('grads',grads.shape, grads)
        
        # print(frame_labels.shape)
        ans.append(find_best_gradients(grads, frame_labels))
        
    return np.array(ans)
    

    # return np.array(find_most_stable_points_per_cluster(G, clustered_frames))


    # for lnd, frame_clusters in zip(LND, clustered_frames):
    #     labels = np.unique(frame_clusters)
        
    #     for label in labels:
    #         pts = [lnd[i]  for i in range(len(lnd)) if frame_clusters[i]==label]





def compute_gradients(landmarks):
    """
    Вычисление градиентов для списка лендмарков.

    Параметры:
    - landmarks: список лендмарков для каждого кадра

    Возвращает:
    - gradients: векторы градиента для каждой точки
    """

    # Преобразуем лендмарки в массив numpy
    landmarks_array = np.array(landmarks)

    # Вычислим градиенты по времени
    gradients = np.diff(landmarks_array, axis=0)
    # print(gradients.shape)
    gradients = np.insert(gradients, 0, gradients[0], axis=0)
    # print(gradients.shape)
    return gradients




def compute_gradients_inhomogeneous(landmarks):
    gradients = []
    for i in range(1, len(landmarks)):
        frame_gradients = []
        for j in range(len(landmarks[i])):
            dx = landmarks[i][j][0] - landmarks[i-1][j][0]
            dy = landmarks[i][j][1] - landmarks[i-1][j][1]
            frame_gradients.append((dx, dy))
        gradients.append(frame_gradients)
    gradients.insert(0, gradients[0])
    return gradients


def cluster_landmarks_by_gradient(landmarks, threshold=0.5):
    gradients = compute_gradients(landmarks)
    # gradients = compute_gradients_inhomogeneous(landmarks)
    movement_intensity = np.linalg.norm(gradients, axis=-1)
    # print(movement_intensity.shape)
    CLST = []
    for frame_lnd in movement_intensity:
        kmeans = KMeans(n_clusters=2, random_state=0)
        clusters = kmeans.fit_predict(frame_lnd.reshape(-1, 1))
        cluster_means = [frame_lnd[clusters == i].mean() for i in range(2)]
        low_intensity_cluster = np.argmin(cluster_means)
        clusters = np.where(clusters == low_intensity_cluster, 0, 1)
        CLST.append(clusters)
    return CLST



def cluster_stable_landmarks(LND, CLST, num_clusters=6):

    STABLE_CLST = []
    for frame_lnd, clst in zip(LND, CLST):
        indicies = [i for i in range(len(clst)) if clst[i] == 0 ]
        stable_on_frame = frame_lnd[indicies]
        # print(stable_on_frame.shape)
        try:
            kmeans_stable = KMeans(n_clusters=num_clusters, random_state=0)
            stable_clusters = kmeans_stable.fit_predict(stable_on_frame.reshape(-1, 2))
            centroids = kmeans_stable.cluster_centers_
            ordered_indices = np.argsort(np.linalg.norm(centroids, axis=1))
            new_stable_clusters = np.zeros_like(stable_clusters)
            for new_idx, old_idx in enumerate(ordered_indices):
                new_stable_clusters[stable_clusters == old_idx] = new_idx

            STABLE_CLST.append([stable_clusters, indicies])
        except:
            STABLE_CLST.append(STABLE_CLST[-1])
    return STABLE_CLST




########################
    ''' UTILS '''
########################

def save_frames_to_video(frames, output_path, fps=25):
    height, width, layers = frames[0].shape
    size = (width, height)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for frame in frames:
        out.write(frame)
    out.release()

def draw_landmarks_on_frames(frames, landmarks_list, stable=None):
    colors_bgr = {
        0: (255, 0, 0),    # Красный
        1: (0, 165, 255),    # orange
        2: (0, 0, 255),    # Синий
        3: (255, 255, 0),  # Желтый
        4: (255, 0, 255),  # Пурпурный
        5: (0, 255, 255)   # Бирюзовый
    }
    frames_with_landmarks = []
    idx = 0
    for frame, landmarks in zip(frames, landmarks_list):
        frame_copy = frame.copy()


        for label in stable:
            for (x, y) in landmarks[stable[label][idx]]:
                cv2.circle(frame_copy, (int(x), int(y)), 2, colors_bgr[label], -1)
        # for (x, y) in landmarks:
            
        #     cv2.circle(frame_copy, (x, y), 2, (0, 255, 0), -1)
        frames_with_landmarks.append(frame_copy)

        idx += 1
    return frames_with_landmarks
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




def filter_points_above_nose(LND):

    n_frames, n_points, _ = LND.shape
    filtered_frames = []

    for i in range(n_frames):
        frame = LND[i]
        nose_y = frame[1, 1]  # y-координата точки носа (индекс 1)
        points_above_nose = frame[frame[:, 1] > nose_y]  # Выбираем только те точки, у которых y > y носа
        filtered_frames.append(points_above_nose)
    print('ABOVE', len(filtered_frames), len(filtered_frames[0]), len(filtered_frames[1]))
    return filtered_frames


def filter_from_chin(LND):
    NEW_LND = []
    for frame_lnd in LND:
        center_chin = frame_lnd[152]
        center_nose = frame_lnd[4]
        radius = np.linalg.norm(center_chin - center_nose)
        distances = np.linalg.norm(frame_lnd - center_chin, axis=1)
        filtered_lnd = frame_lnd[distances > radius]
        NEW_LND.append(filtered_lnd)
    return NEW_LND



def filter_from_chin_soft(LND, mirror_LND=None):

    outside_indices = set()
    for frame_lnd in LND:
        center_chin = frame_lnd[152]
        center_nose = frame_lnd[4]
        radius = np.linalg.norm(center_chin - center_nose)
        distances = np.linalg.norm(frame_lnd - center_chin, axis=1)
        for idx, distance in enumerate(distances):
            if distance > radius:
                outside_indices.add(idx)
    outside_indices = sorted(list(outside_indices))
    NEW_LND = []
    if mirror_LND is None:
        for frame_lnd in LND:
            filtered_lnd = frame_lnd[outside_indices]
            NEW_LND.append(filtered_lnd)

        return NEW_LND
    
    else:
        mirror_NEW_LND = []
        for frame_lnd, mirror_frame_lnd in zip(LND, mirror_LND):
            m_filtered_lnd = mirror_frame_lnd[outside_indices]
            mirror_NEW_LND.append(m_filtered_lnd)

        return mirror_NEW_LND
    
    

def create_label_dict(LND, STABLE_CLST):
    STABLE_LND = {i: [] for i in range(6)}

    
  


    for frame_lnd, stable_clst in zip(LND, STABLE_CLST):
        for label in range(6):
            frame_cluster = []
            
            for j, i in enumerate(stable_clst[1]):
                if stable_clst[0][j] == label:
                    frame_cluster.append(i)


            STABLE_LND[label].append(frame_cluster)
            

    return STABLE_LND


def final_cut(LND, STABLE_LND):
    CUTS = []
    for i, lnd in enumerate(LND):
        data = {i : [] for i in range(6)}
        for label in STABLE_LND:
            try:
                lnd = np.array(lnd)
                current_lnd = lnd[STABLE_LND[label][i]]
                kmeans_stable = KMeans(n_clusters=4, random_state=0)
                stable_clusters = kmeans_stable.fit_predict(current_lnd.reshape(-1, 2))
                data[label] = np.array([[int(stable_clusters[j]), STABLE_LND[label][i][j]] for j in range(len(stable_clusters))])
            except:
                # data[label] = CUTS[-1][label]
                data[label] = []
        CUTS.append(data)

    return CUTS

def get_clustered_landmarks(LND):
    # LND = filter_from_chin_soft(LND)
    # print(np.array(LND).shape)
    CLST = cluster_landmarks_by_gradient(LND)
    # stable_landmarks = np.array([LND[i][CLST[i]] for i in range(len(LND))])
    # print(stable_landmarks.shape)
    # print(CLST[0])
    STABLE_CLST = cluster_stable_landmarks(LND, CLST)
    STABLE_LND = create_label_dict(LND, STABLE_CLST)

    CUTS = final_cut(LND, STABLE_LND)

    return CUTS, LND


   
if __name__ == "__main__":
    LND = load_landmarks('data/joe_face_lnd.csv')
    CUTS = get_clustered_landmarks(LND)

    import json
    # with open('test_data/STABLE.json', 'w', encoding='utf-8') as file:
    #     # STBL = {key : [element.tolist() for element in value ] for (key, value) in STABLE_LND.items()}
    #     json.dump(STABLE_LND, file)
        # file.write(str(STABLE_LND))
    # CUTS = final_cut(LND, STABLE_LND)

    with open('test_data/CUTS.txt', 'w', encoding='utf-8') as file:
        # STBL = {key : [element.tolist() for element in value ] for (key, value) in STABLE_LND.items()}
        file.write(str(CUTS))
    with open('test_data/CUTS.json', 'w', encoding='utf-8') as file:
        # STBL = {key : [element.tolist() for element in value ] for (key, value) in STABLE_LND.items()}
        json.dump(CUTS, file)
    # gen_imgs = get_video('data/joe.mp4')

    import glob

    frames = [cv2.imread(f) for f in glob.glob('data/joe_face/'+'*')]
    print('FRAMES LEN: ', len(frames))
    
    frames_with_landmarks_1 = draw_landmarks_on_frames(frames, LND, stable=STABLE_LND)
    save_frames_to_video(frames_with_landmarks_1, 'test_data/joe_clst.mp4')
    














    # num_clusters = 6
    # XS = np.array([LND[:, i, 0] for i in range(len(LND[0]))])
    # YS = np.array([LND[:, i, 1] for i in range(len(LND[0]))])
    
    # # Кластеризация лендмарок на каждом кадре видео
    # clustered_frames = cluster_landmarks(XS, YS, num_clusters)
    # # labels = np.unique(clustered_frames)

    # stable_points = find_most_stable_points(XS, YS, clustered_frames)
    # print(stable_points.shape)
    # print(stable_points[:50])

    # G = []

    # XS = [LND[:, i, 0] for i in range(len(LND[0]))]
    # YS = [LND[:, i, 1] for i in range(len(LND[0]))]

    # for xs in tqdm(XS):
    #     G.append(lucas_kanade_1d(xs))
    
    # print(np.array(G).shape)
    # print(G[0])

    # for frame_idx in range(3):
    #     print(f"Кадр {frame_idx + 1} стабильные точки:")
    #     for point in stable_points[frame_idx]:
    #         print(f"  Координаты: {point}")