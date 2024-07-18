## hist match 

### usage
---
#### 1. Для метчинга картинок (based on skimage)

####  numpy and skimage required
```{python}
from filter_utils import match_histograms_moded

...

matched_img = match_histograms_moded(generated_image, ref_img, strength=1.0, axis=-1)
# strength = 1.0 for full match, 0.0 for *no* match
# axis = postion of channels in images ; defualt -1 :  ( 512 , 512, 3 )
```
---
#### 2. Для метчинга лиц 
```{python}
from filter_utils import apply_histogram_matching

...

matched_img = apply_histogram_matching(generated_image, ref_img, match_strength=1.0)
# match_strength = 1.0 for full match, 0.0 for *no* match
# эта версия вычисляет лендмарки лица, метчинг более точный для лиц
```

#### 3. Для метчинга лиц pre_exist 
```{python}
from filter_utils import apply_histogram_matching_pre_exist

...

matched_img = apply_histogram_matching_pre_exist(gen_img, drive_img, gen_landmarks, drive_landmarks, match_strength=0.7)
# match_strength = 1.0 for full match, 0.0 for *no* match
# эта версия использует существубщие лендмарки лица, метчинг более точный для лиц.
```
---
### denoise filter 
### usage
#### 1. Для сглаживания лендмарок
```{python}
from utils.euro_filter import smooth_lnd_for_video 

...

smoothed_landmarks = smooth_lnd_for_video(frames_name, landmarks, power = 3, fps=25.0, anchors = None, indicies = None)
# frames_name - список имен видео
# landmarks - путь на файл с лендмарками 
# power - максимальная сила сглаживающего фильтра
# fps - фпс видео
# indicies - индексы лендмарок, которые должны быть отфильтрованы 
# anchors - список якорных координат [ [(x1, y1), (x2, y2), (x3, y3)], [ ... ], ... ]
# если указан список якорных координат, то использование:
smoothed_landmarks, smooth_anchors = smooth_lnd_for_video(frames_name, landmarks, power = 3, fps=25.0, anchors = anchors, indicies = None)
```
csv с лендмарками имеет вид 
x_0, x_1, ... , x_n, y_0, y_1, ..., y_n



#### 2. Для сглаживания лендмарок Kalman
```{python}
from utils.kalman_filter import klmn_filter

...

smoothed_landmarks = klmn_filter(landmarks, POWER = 2, indicies = [i for i in range(len(landmarks[0]))])

# landmarks - список списка лендмарков лендмарков для каждого кадра;  (n, qnt_L, 2) n - кол-во кадров, qnt_L - количество лендмарок,
# power - сила сглаживающего фильтра
# indicies - индексы лендмарок, которые должны быть отфильтрованы 

# Возвращает сглаженные лендмарки  (n, qnt_L, 2)

```


