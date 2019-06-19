from keras import applications
from keras.applications.xception import preprocess_input
from keras.layers import Flatten, Dense, GlobalAveragePooling2D
from keras.models import Sequential, Model, load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.optimizers import SGD, RMSprop
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import numpy as np
import matplotlib.pyplot as plt
import collections
import os
from glob import glob
import numpy as np
import os
import pandas as pd
import shutil
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
def train_model(i):
# parameters
num_classes = 4
postfix = 'KFold_{}'.format(i)
epochs = 20
batch_size = 16
img_width, img_height = 299, 299
# read the file names of training and validating images
train_data_dir = 'data/training'
test_data_dir = 'data/test'
top_weights_path_check = os.path.join(os.getcwd(), '{}.h5'.format(postfix))
# build the Xception networkbase_model = applications.Xception(input_shape=(img_width, img_height, 3),
weights='imagenet',
include_top=False) # Top Model Block
x = Flatten(name='flatten')(base_model.output)
predictions = Dense(num_classes, activation='softmax', name='predictions')(x)
new_model = Model(base_model.input, predictions)
# first: train only the top layers (which were randomly initialized)
# i.e. freeze all layers of the based model that is already pre-trained.
# for layer in base_model.layers:
# layer.trainable = False
new_model.summary()
def get_class_weights(y, smooth_factor=0):
counter = collections.Counter(y)
if smooth_factor > 0:
p = max(counter.values()) * smooth_factor
for k in counter.keys():
counter[k] += p
majority = max(counter.values())
return {cls: float(majority) / count for cls, count in counter.items()}
# prepare data augmentation configuration
# https://stackoverflow.com/questions/42443936/keras-split-train-test-set-when-usingimagedatagenerator
train_datagen = ImageDataGenerator(
preprocessing_function=preprocess_input,
horizontal_flip=True,
rotation_range=10)
train_generator = train_datagen.flow_from_directory(
train_data_dir,
target_size=(img_height, img_width),
batch_size=batch_size,
class_mode='categorical', shuffle=True)
validation_generator = train_datagen.flow_from_directory(
test_data_dir,
target_size=(img_height, img_width),
batch_size=batch_size,
class_mode='categorical', shuffle=False)
sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
new_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=
['accuracy'])
callbacks_list = [ModelCheckpoint(top_weights_path_check, monitor='val_acc',
verbose=1, save_best_only=True)]class_weight = get_class_weights(train_generator.classes)
H = new_model.fit_generator(
train_generator,
epochs=epochs,
class_weight=class_weight,
validation_data=validation_generator,
callbacks=callbacks_list,
verbose=1,
steps_per_epoch=train_generator.samples // batch_size,
validation_steps=validation_generator.samples // batch_size
) n
ew_model.save(top_weights_path_check)
with open('train_hist_{}.pickle'.format(postfix), 'wb') as file_pi:
pickle.dump(H.history, file_pi)
# Confution Matrix and Classification Report
print(H.history.keys())
# summarize history for accuracy
plt.figure(dpi=400)
plt.plot(H.history['acc'], 'b')
plt.plot(H.history['val_acc'], 'c')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('accuracy_{}'.format(postfix))
plt.show()
# summarize history for loss
plt.figure(dpi=400)
plt.plot(H.history['loss'], 'b')
plt.plot(H.history['val_loss'], 'c')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss{}'.format(postfix))
plt.show()
return H.history['val_acc'][-1], H.history['val_loss'][-1]
# used to copy files according to each fold
def copy_images(df, directory):
destination_directory = "data/" + directory
print("copying {} files to {}...".format(directory, destination_directory))
# remove all files from previous fold
if os.path.exists(destination_directory):
shutil.rmtree(destination_directory)
# create folder for files from this fold
if not os.path.exists(destination_directory):os.makedirs(destination_directory)
# create subfolders for each class
for c in set(list(df['class'])):
if not os.path.exists(destination_directory + '/' + c):
os.makedirs(destination_directory + '/' + c)
# copy files for this fold from a directory holding all the files
for i, row in df.iterrows():
try:
# this is the path to all of your images kept together in a separate folder
path_from = "{}"
path_to = "{}/{}".format(destination_directory, row['class'])
# move from folder keeping all files to training, test, or validation folder
(the "directory" argument)
shutil.copy(path_from.format(row['filename']), path_to)
except Exception:
print("Error when copying {}:".format(row['filename']))
# dataframe containing the filenames of the images (e.g., GUID filenames) and the classes
pths = glob('../train_val_images_multiclass/*/*')
# pths = np.array(glob('../train_val_images_multiclass/*/*'))
# pths = list(pths[np.random.choice(len(pths), 500, replace=False)])
df_y = pd.DataFrame([pth.split('/')[-2] for pth in pths], columns=['class'])
df_x = pd.DataFrame(pths, columns=['filename'])
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds)
total_actual = []
total_predicted = []
total_val_accuracy = []
total_val_loss = []
total_test_accuracy = []
for i, (train_index, test_index) in enumerate(skf.split(df_x, df_y)):
x_train, x_test = df_x.iloc[train_index], df_x.iloc[test_index]
y_train, y_test = df_y.iloc[train_index], df_y.iloc[test_index]
train = pd.concat([x_train, y_train], axis=1)
test = pd.concat([x_test, y_test], axis = 1)
# copy the images according to the fold
copy_images(train, 'training')
copy_images(test, 'test')
print('**** Running fold '+ str(i))
# here you call a function to create and train your model, returning validation
accuracy and validation loss
val_accuracy, val_loss = train_model(i)Running model on the filtered dataset to generate predictions
# append validation accuracy and loss for average calculation later on
total_val_accuracy.append(val_accuracy)
total_val_loss.append(val_loss)
# summarize history for kfold
plt.figure(dpi=400)
plt.plot(total_val_accuracy, 'b')
plt.plot([sum(total_val_accuracy)/len(total_val_accuracy)]*len(total_val_accuracy), 'c')
plt.title('{}-Fold Validation Accuracy'.format(n_folds))
plt.ylabel('Accuracy')
plt.xlabel('Folds')
plt.legend(['Accuracy', 'Average Accuracy'], loc='upper left')
plt.savefig('{}_folds'.format(n_folds))
plt.show()
print(classification_report(total_actual, total_predicted))
print(confusion_matrix(total_actual, total_predicted))
print("Validation accuracy on each fold:")
print(total_val_accuracy)
print("Mean validation accuracy: {}%".format(np.mean(total_val_accuracy) * 100))
print("Validation loss on each fold:")
print(total_val_loss)
print("Mean validation loss: {}".format(np.mean(total_val_loss)))
print("Test accuracy on each fold:")
print(total_test_accuracy)
print("Mean test accuracy: {}%".format(np.mean(total_test_accuracy) * 100))
