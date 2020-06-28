#!/usr/bin/env python
# coding: utf-8


import os
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import keras
import pandas as pd
from PIL import Image
import cv2

from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from keras.layers.pooling import MaxPooling2D,AveragePooling2D
from keras.applications.imagenet_utils import decode_predictions

import keras.callbacks as callbacks
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.models import Sequential, Model

import efficientnet.keras as efn 
from efficientnet.keras import center_crop_and_resize, preprocess_input


# In[2]:


def plot_loss_acc(history):
    plt.figure(figsize=(20,7))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'][1:])    
    plt.plot(history.history['val_loss'][1:])    
    plt.title('model loss')    
    plt.ylabel('val_loss')    
    plt.xlabel('epoch')    
    plt.legend(['Train','Validation'], loc='upper left')
    
    plt.subplot(1,2,2)
    plt.plot(history.history['acc'][1:])
    plt.plot(history.history['val_acc'][1:])
    plt.title('Model Accuracy')
    plt.ylabel('val_acc')
    plt.xlabel('epoch')
    plt.legend(['Train','Validation'], loc='upper left')
    plt.show()

    
    
class SnapshotCallbackBuilder:
    def __init__(self, nb_epochs, nb_snapshots, init_lr=0.1):
        self.T = nb_epochs
        self.M = nb_snapshots
        self.alpha_zero = init_lr

    def get_callbacks(self, model_prefix='Model'):

        callback_list = [
            swa,
            callbacks.LearningRateScheduler(schedule=self._cosine_anneal_schedule),
            callback.ModelCheckpoint("./checkpoints/checkpoint.hdf5", monitor='loss', verbose=1,
            save_best_only=True, mode='auto', period=1, save_weights_only=False)
        ]

        return callback_list

    def _cosine_anneal_schedule(self, t):
        cos_inner = np.pi * (t % (self.T // self.M))  # t - 1 is used when t has 1-based indexing.
        cos_inner /= self.T // self.M
        cos_out = np.cos(cos_inner) + 1
        return float(self.alpha_zero / 2 * cos_out)
    
class SWA(keras.callbacks.Callback):
    
    def __init__(self, filepath, swa_epoch):
        super(SWA, self).__init__()
        self.filepath = filepath
        self.swa_epoch = swa_epoch 
    
    def on_train_begin(self, logs=None):
        self.nb_epoch = self.params['epochs']
        print('Stochastic weight averaging selected for last {} epochs.'
              .format(self.nb_epoch - self.swa_epoch))
        
    def on_epoch_end(self, epoch, logs=None):
        
        if epoch == self.swa_epoch:
            self.swa_weights = self.model.get_weights()
            
        elif epoch > self.swa_epoch:    
            for i in range(len(self.swa_weights)):
                self.swa_weights[i] = (self.swa_weights[i] * 
                    (epoch - self.swa_epoch) + self.model.get_weights()[i])/((epoch - self.swa_epoch)  + 1)  

        else:
            pass
        
    def on_train_end(self, logs=None):
        self.model.set_weights(self.swa_weights)
        print('Final model parameters set to stochastic weight average.')
        self.model.save_weights(self.filepath)
        print('Final stochastic averaged weights saved to file.')
def build_finetune_model(base_model, dropout, num_classes):

    x = base_model.output
    
    x = AveragePooling2D((5, 5), name='avg_pool')(x)
    x = Flatten()(x)
    x = Dropout(dropout)(x)
    predictions = Dense(num_classes, activation='softmax', name='finalfc')(x)
    
    finetune_model = Model(inputs=base_model.input, outputs=predictions)

    return finetune_model


# In[3]:


HEIGHT = 299
WIDTH = 299
TARGET_SIZE = (HEIGHT, WIDTH)
BS = 8

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        rotation_range=20.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=[0.9, 1.25],
        brightness_range=[0.5, 1.5],
        validation_split=.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
TRAIN_FOLDER = "./dataset/train/"
test_df = pd.read_csv('./dataset/test.csv')
test_df['category'] = test_df['category'].apply(lambda x: str(x))
test_df['filename'] = test_df['filename'].apply(lambda x: 'test/'+str(x))

train_generator = train_datagen.flow_from_directory(
        TRAIN_FOLDER, 
        target_size=TARGET_SIZE,
        batch_size=BS,
        subset='training', 
        class_mode='categorical')

validation_generator = train_datagen.flow_from_directory(
        TRAIN_FOLDER, 
        target_size=TARGET_SIZE,
        batch_size=BS,
        subset='validation', 
        class_mode='categorical')

test_generator = test_datagen.flow_from_dataframe(
         dataframe=test_df,
         directory='./dataset/',
         target_size=TARGET_SIZE,
         x_col='filename', 
         y_col='category', 
         batch_size=BS,
         class_mode='categorical')


# In[4]:


input_shape=(HEIGHT, WIDTH, 3) # rgb aja, a nya ilangin

dropout = 0.4
epochs = 200
swa = SWA('./keras_swa.model',epochs-3)

base_model = efn.EfficientNetB3(weights='imagenet',
                            include_top=False,
                            input_shape=(HEIGHT, WIDTH, 3))

finetune_model = build_finetune_model(base_model, 
                                      dropout=dropout, 
                                      num_classes=42)

finetune_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

snapshot = SnapshotCallbackBuilder(nb_epochs=epochs,nb_snapshots=1,init_lr=1e-3)

history = finetune_model.fit_generator(generator=train_generator,
                                        validation_data=validation_generator,
                                        steps_per_epoch=train_generator.samples//BS,
                                        epochs=epochs,
                                       verbose=2,
                                       validation_steps=validation_generator.samples//BS,
                                       callbacks=snapshot.get_callbacks())

try:
    finetune_model.load_weights('./keras_swa.model')
except Exception as e:
    print(e)


# In[7]:


swa = SWA('./keras_swab5.model',epochs-3)

base_model = efn.EfficientNetB5(weights='imagenet',
                            include_top=False,
                            input_shape=(HEIGHT, WIDTH, 3))

finetune_model = build_finetune_model(base_model, 
                                      dropout=dropout, 
                                      num_classes=42)

finetune_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

snapshot = SnapshotCallbackBuilder(nb_epochs=epochs,nb_snapshots=1,init_lr=1e-3)

history = finetune_model.fit_generator(generator=train_generator,
                                        validation_data=validation_generator,
                                        steps_per_epoch=train_generator.samples//BS,
                                        epochs=epochs,
                                       verbose=2,
                                       validation_steps=55,
                                       callbacks=snapshot.get_callbacks())

try:
    finetune_model.load_weights('./keras_swab5.model')
except Exception as e:
    print(e)
preds = model.predict_generator(
    test_generator,
    steps=len(test_generator.filenames)
)
image_ids = [name.split('/')[-1] for name in test_generator.filenames]
predictions = preds.flatten()
data = {'id': image_ids, 'has_cactus':predictions} 
submission = pd.DataFrame(data)
print(submission.head())
submission.to_csv('submission.csv', index=False)

# In[8]:


swa = SWA('./keras_swab7.model',epochs-3)

base_model = efn.EfficientNetB7(weights='imagenet',
                            include_top=False,
                            input_shape=(HEIGHT, WIDTH, 3))

finetune_model = build_finetune_model(base_model, 
                                      dropout=0.4, 
                                      num_classes=42)

finetune_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

snapshot = SnapshotCallbackBuilder(nb_epochs=epochs,nb_snapshots=1,init_lr=1e-3)

history = finetune_model.fit_generator(generator=train_generator,
                                        steps_per_epoch=train_generator.samples//BS,
                                        epochs=epochs,
                                       verbose=2,
                                       validation_steps=55,
                                       callbacks=snapshot.get_callbacks())

try:
    finetune_model.load_weights('./keras_swab7.model')
except Exception as e:
    print(e)


# In[ ]:


plot_loss_acc(history)


# In[ ]:




