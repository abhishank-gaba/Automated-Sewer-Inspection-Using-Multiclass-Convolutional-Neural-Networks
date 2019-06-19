from keras import applications
from keras.applications.xception import preprocess_input
from keras.layers import Flatten, Dense, GlobalAveragePooling2D
from keras.models import Sequential, Model, load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.optimizers import SGD, RMSprop
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matriximport pickle
import numpy as np
import matplotlib.pyplot as plt
import collections
import os
# parameters
num_classes = 4
postfix = 'BS2'
epochs = 20
batch_size = 2
img_width, img_height = 299, 299
# read the file names of training and validating images
data_dir = '../train_val_images_multiclass'
validation_split = 0.2
top_weights_path_check = os.path.join(os.getcwd(), '{}.h5'.format(postfix))
# build the Xception network
base_model = applications.Xception(input_shape=(img_width, img_height, 3),
weights='imagenet',
include_top=False) # Top Model Block
x = Flatten(name='flatten')(base_model.output)
# x = Dense(1024, activation='relu', name='fc1')(x)
# x = Dense(1024, activation='relu', name='fc2')(x)
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
# https://stackoverflow.com/questions/42443936/keras-split-train-test-set-when-usingimagedatageneratortrain_datagen = ImageDataGenerator(
preprocessing_function=preprocess_input,
horizontal_flip=True,
rotation_range=10,
validation_split=validation_split)
train_generator = train_datagen.flow_from_directory(
data_dir,
target_size=(img_height, img_width),
batch_size=batch_size,
class_mode='categorical', shuffle=True)
validation_generator = train_datagen.flow_from_directory(
data_dir,
target_size=(img_height, img_width),
batch_size=batch_size,
class_mode='categorical', shuffle=False,
subset='validation')
sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
new_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
callbacks_list = [ModelCheckpoint(top_weights_path_check, monitor='val_acc', verbose=1,
save_best_only=True)]
class_weight = get_class_weights(train_generator.classes)
H = new_model.fit_generator(
train_generator,
epochs=epochs,
# class_weight=class_weight,
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
plt.plot(H.history['acc'],'b')
plt.plot(H.history['val_acc'],'c')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')Training model using K-Fold Validation
plt.savefig('accuracy_{}'.format(postfix))
plt.show()
# summarize history for loss
plt.figure(dpi=400)
plt.plot(H.history['loss'],'b')
plt.plot(H.history['val_loss'],'c')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss{}'.format(postfix))
plt.show()
