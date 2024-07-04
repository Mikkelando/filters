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
from filter_utils import smooth_lnd_for_video

...

smoothed_landmarks = smooth_lnd_for_video(frames_name, landmarks, power = 3, fps=25.0, qnt_l = 468)
# frames_name - список имен видео
# landmarks - список лендмарок для каждого кадра (список списков)
# power - максимальная сила сглаживающего фильтра
# fps - фпс видео
# qnt_l -количество лендмарок (468, 68)
```