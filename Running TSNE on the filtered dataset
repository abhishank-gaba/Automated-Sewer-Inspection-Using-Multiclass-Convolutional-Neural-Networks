from sklearn.manifold import TSNE
from glob import glob
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import pickle
import ast
def parse_tuple(string):
try:
s = ast.literal_eval(str(string))
if type(s) == tuple:
return s
return
except:
return
# Subset of images for clustering
clust_imgs = []
df = pd.read_csv("../labels.csv", index_col=0, dtype={'filename':str, 'tags': str,
'index': int})
df['tags'] = df['tags'].apply(parse_tuple)
images = glob('../train_val_images_multiclass/*/*')
n_imgs = len(images)
chosen_idx = [np.random.choice(len(images), n_imgs, replace=False)]
types = np.array([image.split('/')[-2] for image in images])[chosen_idx]
images = np.array([image.split('/')[-1] for image in images])
image_paths = images[chosen_idx]
# Import mlb
with open('mlb.pkl', 'rb') as f:
mlb = pickle.load(f)
# Image preprocessing
for i in image_paths:
img = plt.imread("../raw_images/" + i)
img = cv2.resize(img, (100, 100), cv2.INTER_LINEAR).astype('float')
# img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype('float')
img = cv2.normalize(img, None, 0.0, 1.0, cv2.NORM_MINMAX)
img = img.reshape(1, -1)
clust_imgs.append(img)
# Convert into a Numpy array
img_mat = np.vstack(clust_imgs)
# Number of images, (100pix by 100pix) by 4 bands
print(img_mat.shape)
# Fit a t-SNE manifold to the subset of images
tsne = TSNE(
n_components=2,Training model for parametrization analysis
init='random', # pca
perplexity=30,
random_state=101,
method='barnes_hut',
n_iter=6000,
verbose=2
).fit_transform(img_mat)
# Plot the subset of images in a two dimensional representation
def imscatter(x, y, images, ax=None, zoom=0.1):
ax = plt.gca()
images = [OffsetImage(image, zoom=zoom) for image in images]
artists = []
for x0, y0, im0 in zip(x, y, images):
ab = AnnotationBbox(im0, (x0, y0), xycoords='data', frameon=False)
artists.append(ax.add_artist(ab))
ax.update_datalim(np.column_stack([x, y]))
ax.autoscale()
plt.figure(figsize=(20, 20))
images = [plt.imread("../raw_images/"+image_paths[i]) for i in range(n_imgs)]
for i in range(n_imgs):
tg = df[df.filename == image_paths[i]].tags.values[0]
if types[i] == 'none': # none
images[i] = cv2.rectangle(images[i], (0, 0), (50, 50), (255, 0, 0), -1)
elif types[i] == 'connect': # connection
images[i] = cv2.rectangle(images[i], (0, 0), (50, 50), (0, 255, 0), -1)
elif types[i] == 'joint': # joint
images[i] = cv2.rectangle(images[i], (0, 0), (50, 50), (0, 0, 255), -1)
elif types[i] == 'manhole': # manhole
images[i] = cv2.rectangle(images[i], (0, 0), (50, 50), (255, 0, 255), -1)
else:
raise ValueError("Error")
images = [cv2.resize(image, (300, 300), cv2.INTER_LINEAR).astype('int64') for image in
images]
imscatter(tsne[0:n_imgs, 0], tsne[0:n_imgs, 1], images)
plt.savefig("tSNE.jpg")
