from glob import glob
from os import remove
import numpy as np
n_imgs = 1000
files = glob("../train_val_images_multiclass/joint/*")
chosen_idx = [np.random.choice(len(files), n_imgs, replace=False)]
files = np.array(files)[chosen_idx]
for f in files:
remove(f)
