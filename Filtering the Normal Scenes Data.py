from glob import glob
from shutil import move
import numpy as np
n_imgs = 32000
files = glob("../train_val_images_connect/none/*")
chosen_idx = [np.random.choice(len(files), n_imgs, replace=False)]
files = np.array(files)[chosen_idx]
for f in files:
move(f, "../train_val_images_connect_extra/none/")
