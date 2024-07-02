## hist match (based on skimage)

####  numpy and skimage required
### usage
---
#### 1. Для метчинга картинок 
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

matched_img = apply_histogram_matching(generated_image, ref_img, strength=1.0)
# strength = 1.0 for full match, 0.0 for *no* match
# эта версия вычисляет лендмарки лица, метчинг более точный для лиц
```