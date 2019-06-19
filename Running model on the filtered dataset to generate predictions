from glob import glob
import cv2, os, shutil
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.applications.xception import preprocess_input
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from copy import deepcopy
import matplotlib.pyplot as plt
import json
def plot_confusion_matrix(y_true, y_pred, classes,
normalize=False,
title=None,
cmap=plt.cm.Blues):
"""
This function prints and plots the confusion matrix.
Normalization can be applied by setting `normalize=True`.
"""if not title:
if normalize:
title = 'Normalized confusion matrix'
else:
title = 'Confusion matrix, without normalization'
# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)
# Only use the labels that appear in the data
classes = classes[unique_labels(y_true, y_pred)]
if normalize:
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print("Normalized confusion matrix")
else:
print('Confusion matrix, without normalization')
print(cm)
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
ax.figure.colorbar(im, ax=ax)
# We want to show all ticks...
ax.set(xticks=np.arange(cm.shape[1]),
yticks=np.arange(cm.shape[0]),
# ... and label them with the respective list entries
xticklabels=classes, yticklabels=classes,
title=title,
ylabel='True label',
xlabel='Predicted label')
# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
rotation_mode="anchor")
# Loop over data dimensions and create text annotations.
fmt = '.2f' if normalize else 'd'
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
for j in range(cm.shape[1]):
ax.text(j, i, format(cm[i, j], fmt),
ha="center", va="center",
color="white" if cm[i, j] > thresh else "black")
fig.tight_layout()
if normalize:
plt.savefig('confusion_matrix_normalized')
else:
plt.savefig('confusion_matrix')
return ax
# Retrieve data (Videos) list
images = glob('../train_val_images/*/*')
# video_idx = round(random.random() * len(videos))
# print("Video: {}".format(videos[int(video_idx)]))# Model Parameters
top_weights_path_check = os.path.join(os.getcwd(), 'BS2.h5')
xception_height = 299
xception_width = 299
d = {'connect': 0, 'joint': 1, 'manhole': 2, 'none': 3}
target_names = ['connection', 'joint', 'manhole', 'none']
# Load Model
new_model = load_model(top_weights_path_check)
result = []
# images = np.random.choice(images, 1000)
# Run through and predict each video
for image in tqdm(images):
# Video parameters
tag = image.split('/')[-2]
# generate predictions
img = plt.imread(image)
# Pre-process image
tmp = cv2.resize(img, dsize=(xception_height, xception_width),
interpolation=cv2.INTER_LINEAR).astype(np.float64)
tmp = np.expand_dims(tmp, axis=0)
tmp = preprocess_input(tmp)
# Predict and store results
prediction = np.argmax(new_model.predict(tmp))
# Based on prediction, overlay green circle (non-joint) or red circle (joint)
if prediction == 0:
result.append([image.split('/')[-1], 0, d[tag]])
elif prediction == 1:
result.append([image.split('/')[-1], 1, d[tag]])
elif prediction == 2:
result.append([image.split('/')[-1], 2, d[tag]])
elif prediction == 3:
result.append([image.split('/')[-1], 3, d[tag]])
else:
raise ValueError("Something's gone wrong")
# Output predictions to csv and images
predictions = pd.DataFrame(result, columns=['filename', 'predicted', 'actual'])
predictions.to_csv('predictions.csv')
cr = classification_report(predictions.actual.values, predictions.predicted.values,
target_names=target_names)
with open('classification_report', 'w') as f:
json.dump(cr, f)
print(cr)
# Plot non-normalized confusion matrixMoving Classes into their respective folders
plot_confusion_matrix(predictions.actual.values, predictions.predicted.values,
classes=np.array(target_names), title='Confusion matrix, without normalization')
plot_confusion_matrix(predictions.actual.values, predictions.predicted.values,
classes=np.array(target_names), title='Confusion matrix, without normalization',
normalize=True)
