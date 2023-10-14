# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 11:18:56 2023

@author: 532807
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import Xception, VGG16, VGG19
from tensorflow.keras.applications import ResNet50, ResNet101, ResNet152
from tensorflow.keras.applications import ResNet50V2, ResNet101V2, ResNet152V2
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import MobileNet, MobileNetV2, MobileNetV3Small, MobileNetV3Large
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from tensorflow.keras.applications import EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2, EfficientNetV2B3
from tensorflow.keras.applications import EfficientNetV2S, EfficientNetV2M, EfficientNetV2L
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


VERBOSE = 1
EPOCH1 = 5
EPOCH2 = 5

data_dir = 'c:/IA/Data'
batch_size = 32
img_height = 224 #256
img_width = 224 #256

print(tf.config.list_physical_devices('GPU'))

# Load the data

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Configure the dataset for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Create the base model from the pre-trained EfficientNet

base_model = {}
model = {}
model_name = {}
hist = {}

### Xception ###
model_name[0] = "Xception"
base_model[0] = Xception(input_shape=(img_height, img_width, 3),
                            include_top=False,
                            weights='imagenet')

### VGG16 ###
model_name[1] = "VGG16"
base_model[1] = VGG16(input_shape=(img_height, img_width, 3),
                            include_top=False,
                            weights='imagenet')

### VGG19 ###
model_name[2] = "VGG19"
base_model[2] = VGG19(input_shape=(img_height, img_width, 3),
                            include_top=False,
                            weights='imagenet')

### ResNet50 ###
model_name[3] = "ResNet50"
base_model[3] = ResNet50(input_shape=(img_height, img_width, 3),
                            include_top=False,
                            weights='imagenet')

### ResNet101 ###
model_name[4] = "ResNet101"
base_model[4] = ResNet101(input_shape=(img_height, img_width, 3),
                            include_top=False,
                            weights='imagenet')

### ResNet152 ###
model_name[5] = "ResNet152"
base_model[5] = ResNet152(input_shape=(img_height, img_width, 3),
                            include_top=False,
                            weights='imagenet')

### ResNet50 V2 ###
model_name[6] = "ResNet50_V2"
base_model[6] = ResNet50V2(input_shape=(img_height, img_width, 3),
                            include_top=False,
                            weights='imagenet')

### ResNet101 V2 ###
model_name[7] = "ResNet101_V2"
base_model[7] = ResNet101V2(input_shape=(img_height, img_width, 3),
                            include_top=False,
                            weights='imagenet')

### ResNet152 V2 ###
model_name[8] = "ResNet152_V2"
base_model[8] = ResNet152V2(input_shape=(img_height, img_width, 3),
                            include_top=False,
                            weights='imagenet')

### Inception V3 ###
model_name[9] = "Inception_V3"
base_model[9] = InceptionV3(input_shape=(img_height, img_width, 3),
                            include_top=False,
                            weights='imagenet')

### InceptionResNet V2 ###
model_name[10] = "InceptionResNet_V2"
base_model[10] = InceptionResNetV2(input_shape=(img_height, img_width, 3),
                            include_top=False,
                            weights='imagenet')

### MobileNet ###
model_name[11] = "MobileNet"
base_model[11] = MobileNet(input_shape=(img_height, img_width, 3),
                            include_top=False,
                            weights='imagenet')

### MobileNet V2 ###
model_name[12] = "MobileNet_V2"
base_model[12] = MobileNetV2(input_shape=(img_height, img_width, 3),
                            include_top=False,
                            weights='imagenet')

### MobileNet V3 Small ###
model_name[13] = "MobileNet_V3_Small"
base_model[13] = MobileNetV3Small(input_shape=(img_height, img_width, 3),
                            include_top=False,
                            weights='imagenet')

### MobileNet V3 Large ###
model_name[14] = "MobileNet_V3_Large"
base_model[14] = MobileNetV3Large(input_shape=(img_height, img_width, 3),
                            include_top=False,
                            weights='imagenet')

### EfficientNetB0 ###
model_name[15] = "EfficientNet_B0"
base_model[15] = EfficientNetB0(input_shape=(img_height, img_width, 3),
                            include_top=False,
                            weights='imagenet')

### EfficientNetB1 ###
model_name[16] = "EfficientNet_B1"
base_model[16] = EfficientNetB1(input_shape=(img_height, img_width, 3),
                            include_top=False,
                            weights='imagenet')

### EfficientNetB2 ###
model_name[17] = "EfficientNet_B2"
base_model[17] = EfficientNetB2(input_shape=(img_height, img_width, 3),
                            include_top=False,
                            weights='imagenet')

### EfficientNetB3 ###
model_name[18] = "EfficientNet_B3"
base_model[18] = EfficientNetB3(input_shape=(img_height, img_width, 3),
                            include_top=False,
                            weights='imagenet')

### EfficientNetB4 ###
model_name[19] = "EfficientNet_B4"
base_model[19] = EfficientNetB4(input_shape=(img_height, img_width, 3),
                            include_top=False,
                            weights='imagenet')

### EfficientNetB5 ###
model_name[20] = "EfficientNet_B5"
base_model[20] = EfficientNetB5(input_shape=(img_height, img_width, 3),
                            include_top=False,
                            weights='imagenet')

### EfficientNetB6 ###
model_name[21] = "EfficientNet_B6"
base_model[21] = EfficientNetB6(input_shape=(img_height, img_width, 3),
                            include_top=False,
                            weights='imagenet')

### EfficientNetB7 ###
model_name[22] = "EfficientNet_B7"
base_model[22] = EfficientNetB7(input_shape=(img_height, img_width, 3),
                            include_top=False,
                            weights='imagenet')

### EfficientNetB0 V2 ###
model_name[23] = "EfficientNet_B0_V2"
base_model[23] = EfficientNetV2B0(input_shape=(img_height, img_width, 3),
                            include_top=False,
                            weights='imagenet')

### EfficientNetB1 V2 ###
model_name[24] = "EfficientNet_B1_V2"
base_model[24] = EfficientNetV2B1(input_shape=(img_height, img_width, 3),
                            include_top=False,
                            weights='imagenet')

### EfficientNetB2 V2 ###
model_name[25] = "EfficientNet_B2_V2"
base_model[25] = EfficientNetV2B2(input_shape=(img_height, img_width, 3),
                            include_top=False,
                            weights='imagenet')

### EfficientNetB3 V2 ###
model_name[26] = "EfficientNet_B3_V2"
base_model[26] = EfficientNetV2B3(input_shape=(img_height, img_width, 3),
                            include_top=False,
                            weights='imagenet')

### EfficientNet2S ###
model_name[27] = "EfficientNet_V2_Small"
base_model[27] = EfficientNetV2S(input_shape=(img_height, img_width, 3),
                            include_top=False,
                            weights='imagenet')

### EfficientNet2M ###
model_name[28] = "EfficientNet_V2_Medium"
base_model[28] = EfficientNetV2M(input_shape=(img_height, img_width, 3),
                            include_top=False,
                            weights='imagenet')

### EfficientNet2L ###
model_name[29] = "EfficientNet_V2_Large"
base_model[29] = EfficientNetV2L(input_shape=(img_height, img_width, 3),
                            include_top=False,
                            weights='imagenet')

for i in range(30):
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(model_name[i]+".h5", verbose=1, save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir='./logs'),
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ]
    
    # Freeze the base model
    
    print("\nTraining of model " + str(i+1) + " " + model_name[i] + "\n")
    
    base_model[i].trainable = False
    
    # Add a new classifier layers on top of the base model
    inputs = tf.keras.Input(shape=(img_height, img_width, 3))
    x = base_model[i](inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(4)(x) # Assuming you have 4 classes
    model[i] = tf.keras.Model(inputs, outputs)
    
    # Compile the model
    model[i].compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    # Train the model
    hist[i] = model[i].fit(train_ds, validation_data=val_ds, epochs=EPOCH1, callbacks=callbacks)
    
    # Fine-tune the base model
    base_model[i].trainable = True
    
    model[i].compile(optimizer=tf.keras.optimizers.Adam(1e-5), # Low learning rate for fine-tuning
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    hist[i] = model[i].fit(train_ds, validation_data=val_ds, epochs=EPOCH2, callbacks=callbacks)
    
    model_json = model[i].to_json()
    with open(model_name[i]+'.json', 'w') as json_file:
        json_file.write(model_json)
    
    hist_ = pd.DataFramne(hist[i].history)
    hist_

    plt.figure(figsize=(15,5))

    plt.subplot(1,2,1)
    plt.plot(hist_['loss'],label='Train_Loss')
    plt.plot(hist_['val_loss'],label='Validation_Loss')
    plt.title('Train_Loss & Validation_Loss',fontsize=20)
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(hist_['accuracy'],label='Train_Accuracy')
    plt.plot(hist_['val_accuracy'],label='Validation_Accuracy')
    plt.title('Train_Accuracy & Validation_Accuracy',fontsize=20)
    plt.legend()
    
    x_val = np.array(list(val_ds.map(lambda x: x['image'])))
    y_val = np.array(list(val_ds.map(lambda y: y['label'])))
    
    y_pred = []
    predictions=model.predict(np.array(x_val))
    for j in predictions:
        y_pred.append(np.argmax(j))
    df=pd.DataFrame()
    df['Actual'],df['Prediction']=y_val,y_pred
    df

    ax= plt.subplot()
    CM = confusion_matrix(y_val,y_pred)
    sns.heatmap(CM, annot=True, fmt='g', ax=ax,cbar=False,cmap='RdBu')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels') 
    ax.set_title('Confusion Matrix')
    plt.show()
    CM

    ClassificationReport = classification_report(y_val,y_pred)
    print('Classification Report is : ', ClassificationReport )
    
    print ("\n------------------------\n")