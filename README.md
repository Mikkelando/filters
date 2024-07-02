## hist match (based on skimage)

### usage
```{python}
from filter_utils import match_histograms_moded

...

matched_img = match_histograms_moded(generated_image, ref_img, strength=1.0, axis=-1)
# strength = 1.0 for full match, 0.0 for *no* match
# axis = postion of channels in images ; defualt -1 :  ( 512 , 512, 3 )
```
