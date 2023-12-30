"""
Created on Thu Oct 19 18:06:12 2023

@author: UMONS - 532807
"""
import os.path
from pathlib import Path
import platform
import tkinter as tk
from tkinter import ttk
import tensorflow as tf
import keras  as k
# import keras_tuner as kt
from tensorflow.keras import layers
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications import ResNet101V2
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.applications import RegNetX002
from tensorflow.keras.applications import RegNetX004
from tensorflow.keras.applications import RegNetX006
from tensorflow.keras.applications import RegNetX008
from tensorflow.keras.applications import RegNetX016
from tensorflow.keras.applications import RegNetX032
from tensorflow.keras.applications import RegNetX040
from tensorflow.keras.applications import RegNetX064
from tensorflow.keras.applications import RegNetX080
from tensorflow.keras.applications import RegNetX120
from tensorflow.keras.applications import RegNetX160
from tensorflow.keras.applications import RegNetX320
from tensorflow.keras.applications import RegNetY002
from tensorflow.keras.applications import RegNetY004
from tensorflow.keras.applications import RegNetY006
from tensorflow.keras.applications import RegNetY008
from tensorflow.keras.applications import RegNetY016
from tensorflow.keras.applications import RegNetY032
from tensorflow.keras.applications import RegNetY040
from tensorflow.keras.applications import RegNetY064
from tensorflow.keras.applications import RegNetY080
from tensorflow.keras.applications import RegNetY120
from tensorflow.keras.applications import RegNetY160
from tensorflow.keras.applications import RegNetY320
from tensorflow.keras.applications import ResNetRS50
from tensorflow.keras.applications import ResNetRS101
from tensorflow.keras.applications import ResNetRS152
from tensorflow.keras.applications import ResNetRS200
from tensorflow.keras.applications import ResNetRS270
from tensorflow.keras.applications import ResNetRS350
from tensorflow.keras.applications import ResNetRS420
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras.applications import NASNetLarge
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications import EfficientNetB5
from tensorflow.keras.applications import EfficientNetB6
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.applications import EfficientNetV2B1
from tensorflow.keras.applications import EfficientNetV2B2
from tensorflow.keras.applications import EfficientNetV2B3
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.applications import EfficientNetV2M
from tensorflow.keras.applications import EfficientNetV2L
from tensorflow.keras.applications import ConvNeXtTiny
from tensorflow.keras.applications import ConvNeXtSmall
from tensorflow.keras.applications import ConvNeXtBase
from tensorflow.keras.applications import ConvNeXtLarge
from tensorflow.keras.applications import ConvNeXtXLarge
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from sklearn.metrics import confusion_matrix, classification_report
import sklearn as sk
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

numgpu = len(tf.config.list_physical_devices('GPU'))
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

variables = {}
models = {}
 
p = Path()
if platform.system() == "Windows":
    data_dir = str(p) + '\Data'
    output_dir = str(p) + '\Output'  
else :
    data_dir = str(p) + '/Data'
    output_dir = str(p) + '/Output'

root = tk.Tk()
root.title('Ai Toolbox - Deep Learning - Classifier')
root.columnconfigure(0, weight=1)

mc = ttk.Frame(root)
mc.grid(padx=10, 
        pady=10, 
        sticky=tk.W + tk.E)
mc.columnconfigure(0, weight=1)

data_info = ttk.LabelFrame(mc, text='Data Parameters')
data_info.grid(padx=5, 
               pady=5, 
               sticky=tk.W + tk.E)
for i in range(4):
    data_info.columnconfigure(i, weight=1)
 
variables["datapath"] = tk.StringVar()
ttk.Label(data_info, 
          text="Data Path").grid(row=0, 
                                 column=0, 
                                 sticky=tk.W + tk.E, 
                                 padx=5, 
                                 pady=5)
datapath = ttk.Entry(data_info, textvariable=variables["datapath"])
datapath.grid(row=1, 
              columnspan=2, 
              padx=5, 
              pady=5, 
              sticky=tk.W + tk.E)
variables["datapath"].set(data_dir)

variables["outputdata"] = tk.StringVar()
ttk.Label(data_info, 
          text="Output Path").grid(row=0, 
                                   column=3, 
                                   sticky=tk.W + tk.E, 
                                   padx=5, 
                                   pady=5)
outputpath = ttk.Entry(data_info, textvariable=variables["outputdata"])
outputpath.grid(row=1, 
                columnspan=2, 
                column=3, 
                padx=5, 
                pady=5, 
                sticky=tk.W + tk.E)
variables["outputdata"].set(output_dir)

variables["imgresizing"] = tk.StringVar()
ttk.Label(data_info, 
          text="Image Resizing (pixels)").grid(row=2, 
                                               column=0, 
                                               sticky=(tk.W + tk.E), 
                                               padx=5, 
                                               pady=5)
listsize = ['28 x 28', 
            '32 x 32', 
            '64 x 64', 
            '128 x 128', 
            '224 x 224', 
            '256 x 256', 
            '299 x 299', 
            '300 x 300', 
            '400 x 400', 
            '512 x 512']
imgresize = ttk.Combobox(data_info, 
                         values=listsize, 
                         textvariable=variables["imgresizing"], 
                         state='readonly')
imgresize.grid(row=3, 
               column=0, 
               sticky=(tk.W + tk.E), 
               padx=5, 
               pady=5)
imgresize.current(4)

variables["channel"] = tk.IntVar()
ttk.Label(data_info, 
          text="Number of channels").grid(row=2, 
                                          column=1, 
                                          sticky=(tk.W + tk.E), 
                                          padx=5, 
                                          pady=5)
channel = ttk.Spinbox(data_info, 
                      textvariable=variables["channel"], 
                      from_=1, 
                      to=4, 
                      increment=1, 
                      state='readonly')
channel.grid(row=3, 
             column=1, 
             sticky=(tk.W + tk.E), 
             padx=5, 
             pady=5)
channel.set(3)

variables["classes"] = tk.IntVar()
ttk.Label(data_info, 
          text="Number of classes").grid(row=2, 
                                         column=2, 
                                         sticky=(tk.W + tk.E), 
                                         padx=5, 
                                         pady=5)
classes = ttk.Spinbox(data_info, 
                      textvariable=variables["classes"], 
                      from_=2, 
                      to=1000, 
                      increment=1, 
                      state='readonly')
classes.grid(row=3, 
             column=2, 
             sticky=(tk.W + tk.E), 
             padx=5, 
             pady=5)
classes.set(4)

variables["valsplit"] = tk.DoubleVar()
ttk.Label(data_info, 
          text="Validation Split").grid(row=2, 
                                        column=3, 
                                        sticky=(tk.W + tk.E), 
                                        padx=5, 
                                        pady=5)
valsplit = ttk.Spinbox(data_info, 
                       textvariable=variables["valsplit"], 
                       from_=0, 
                       to=1, 
                       increment=0.01, 
                       state='readonly')
valsplit.grid(row=3, 
              column=3, 
              sticky=(tk.W + tk.E), 
              padx=5, 
              pady=5)
valsplit.set(0.2)

variables["batchsize"] = tk.IntVar()
ttk.Label(data_info, 
          text="Batch Size").grid(row=2, 
                                  column=4, 
                                  sticky=(tk.W + tk.E), 
                                  padx=5, 
                                  pady=5)
listbatch = [1,2,4,8,16,32,64,128,256,512]
batchsize = ttk.Combobox(data_info, 
                         values=listbatch, 
                         textvariable=variables["batchsize"], 
                         state='readonly')
batchsize.grid(row=3, 
               column=4, 
               sticky=(tk.W + tk.E), 
               padx=5, 
               pady=5)
batchsize.current(5)



augment_info = ttk.LabelFrame(mc, text='Data Augmentation')
augment_info.grid(padx=5, 
                  pady=5, 
                  sticky=(tk.W + tk.E))
for i in range(8):
    augment_info.columnconfigure(i, weight=1)
   

   
def augment_sel():
    if variables["augment"].get() == 0:
        crop['state'] = 'disabled'
        crop.state(['!selected'])
        horizflip['state'] = 'disabled'
        horizflip.state(['!selected'])
        vertiflip['state'] = 'disabled'
        vertiflip.state(['!selected'])
        translation['state'] = 'disabled'
        translation.state(['selected'])
        rotation['state'] = 'disabled'   
        rotation.state(['!selected'])
        zoom['state'] = 'disabled'
        zoom.state(['!selected'])
        contrast['state'] = 'disabled'
        contrast.state(['!selected'])
        brightness['state'] = 'disabled'
        brightness.state(['!selected'])
    else:
        crop['state'] = 'enabled'
        horizflip['state'] = 'enabled'
        vertiflip['state'] = 'enabled'
        translation['state'] = 'enabled'
        rotation['state'] = 'enabled'        
        zoom['state'] = 'enabled'
        contrast['state'] = 'enabled'
        brightness['state'] = 'enabled'      
        
        
variables["augment"] = tk.IntVar()
augment = ttk.Checkbutton(augment_info, 
                          text="Activate Data Augmentation", 
                          variable=variables["augment"], 
                          command=augment_sel)
augment.grid(row=0, columnspan=8)
augment['state']='disabled'

ttk.Separator(augment_info, orient='horizontal').grid(row=1, 
                                                      columnspan=8, 
                                                      padx=5, 
                                                      pady=5, 
                                                      sticky=(tk.W + tk.E))

variables["crop"] = tk.IntVar() 
crop = ttk.Checkbutton(augment_info, 
                       text="Cropping", 
                       variable=variables["crop"], 
                       onvalue=True, 
                       offvalue=False)
crop.grid(row=2, 
          column=0, 
          sticky=(tk.W + tk.E), 
          padx=5, 
          pady=5)
crop['state'] = 'disabled'

variables['horizflip'] = tk.IntVar()
horizflip = ttk.Checkbutton(augment_info, 
                            text="Horizontal Flip", 
                            variable=variables["horizflip"], 
                            onvalue=True, 
                            offvalue=False)
horizflip.grid(row=2, 
               column=1,
               sticky=(tk.W + tk.E), 
               padx=5, 
               pady=5)
horizflip['state'] = 'disabled'

variables['vertiflip'] = tk.IntVar()
vertiflip = ttk.Checkbutton(augment_info, 
                            text="Verticial Flip", 
                            variable=variables["vertiflip"], 
                            onvalue=True, 
                            offvalue=False)
vertiflip.grid(row=2, 
               column=2, 
               sticky=(tk.W + tk.E), 
               padx=5, 
               pady=5)
vertiflip['state'] = 'disabled'

variables['translation'] = tk.IntVar()
translation = ttk.Checkbutton(augment_info, 
                              text="Translation", 
                              variable=variables["translation"], 
                              onvalue=True, 
                              offvalue=False)
translation.grid(row=2, 
                 column=3, 
                 sticky=(tk.W + tk.E), 
                 padx=5, 
                 pady=5)
translation['state'] = 'disabled'

variables['rotation'] = tk.IntVar()
rotation = ttk.Checkbutton(augment_info,
                           text="Rotation", 
                           variable=variables["rotation"], 
                           onvalue=True, 
                           offvalue=False)
rotation.grid(row=2, 
              column=4, 
              sticky=(tk.W + tk.E), 
              padx=5, 
              pady=5)
rotation['state'] = 'disabled'

variables['zoom'] = tk.IntVar()
zoom = ttk.Checkbutton(augment_info, 
                       text="Zoom", 
                       variable=variables["zoom"], 
                       onvalue=True, 
                       offvalue=False)
zoom.grid(row=2, 
          column=5, 
          sticky=(tk.W + tk.E), 
          padx=5, 
          pady=5)
zoom['state'] = 'disabled'

variables['contrast'] = tk.IntVar()
contrast = ttk.Checkbutton(augment_info, 
                           text="Contrast", 
                           variable=variables["contrast"], 
                           onvalue=True, 
                           offvalue=False)
contrast.grid(row=2, 
              column=6, 
              sticky=(tk.W + tk.E), 
              padx=5, 
              pady=5)
contrast['state'] = 'disabled'

variables['brightness'] = tk.IntVar()
brightness = ttk.Checkbutton(augment_info, 
                             text="Brightness", 
                             variable=variables["brightness"], 
                             onvalue=True, 
                             offvalue=False)
brightness.grid(row=2, 
                column=7, 
                sticky=(tk.W + tk.E), 
                padx=5, 
                pady=5)
brightness['state'] = 'disabled'



mc_info = ttk.LabelFrame(mc, text='Model(s) selection')
mc_info.grid(padx=5, 
             pady=5, 
             sticky=(tk.W + tk.E))
for i in range(8):
    mc_info.columnconfigure(i, weight=1)
    
# Xception       
models['Xception'] = tk.BooleanVar()
model_Xception = ttk.Checkbutton(mc_info, 
                                 text="Xception", 
                                 variable=models['Xception'], 
                                 onvalue=True, 
                                 offvalue=False)
model_Xception.grid(row=1, 
                    column=0, 
                    sticky=(tk.W + tk.E))

# VGG 16
models['VGG16'] = tk.BooleanVar()
model_VGG16 = ttk.Checkbutton(mc_info, 
                              text="VGG16", 
                              variable=models['VGG16'], 
                              onvalue=True, 
                              offvalue=False)
model_VGG16.grid(row=1, 
                 column=1, 
                 sticky=(tk.W + tk.E))

# VGG 19
models['VGG19'] = tk.BooleanVar()
model_VGG19 = ttk.Checkbutton(mc_info, 
                              text="VGG19", 
                              variable=models['VGG19'], 
                              onvalue=True, 
                              offvalue=False)
model_VGG19.grid(row=1, 
                 column=2, 
                 sticky=(tk.W + tk.E))

# ResNet 50
models['ResNet50'] = tk.BooleanVar()
model_ResNet50 = ttk.Checkbutton(mc_info, 
                                 text="ResNet50", 
                                 variable=models['ResNet50'], 
                                 onvalue=True, 
                                 offvalue=False)
model_ResNet50.grid(row=1, 
                    column=3, 
                    sticky=(tk.W + tk.E))

# ResNet 50 Version 2
models['ResNet50V2'] = tk.BooleanVar()
model_ResNet50V2 = ttk.Checkbutton(mc_info, 
                                   text="ResNet50V2", 
                                   variable=models['ResNet50V2'], 
                                   onvalue=True, 
                                   offvalue=False)
model_ResNet50V2.grid(row=1, 
                      column=4, 
                      sticky=(tk.W + tk.E))

# ResNetRS 50 
models["ResNetRS50"] = tk.BooleanVar()
model_ResNetRS50 = ttk.Checkbutton(mc_info, 
                                   text='ResNetRS50', 
                                   variable=models['ResNetRS50'], 
                                   onvalue=True, 
                                   offvalue=False)
model_ResNetRS50.grid(row=1, 
                      column=5, 
                      sticky=(tk.W + tk.E))

# ResNet 101
models['ResNet101'] = tk.BooleanVar()
model_ResNet101 = ttk.Checkbutton(mc_info, 
                                  text="ResNet101", 
                                  variable=models["ResNet101"], 
                                  onvalue=True, 
                                  offvalue=False)
model_ResNet101.grid(row=1, 
                     column=6, 
                     sticky=(tk.W + tk.E))

# ResNet 101 Version 2
models["ResNet101V2"] = tk.BooleanVar()
model_ResNet101V2 = ttk.Checkbutton(mc_info, 
                                    text="ResNet101V2", 
                                    variable=models["ResNet101V2"], 
                                    onvalue=True, 
                                    offvalue=False)
model_ResNet101V2.grid(row=1, 
                       column=7, 
                       sticky=(tk.W + tk.E))



# ResNetRS 101
models["ResNetRS101"] = tk.BooleanVar()
model_ResNetRS101 = ttk.Checkbutton(mc_info, 
                                    text='ResNetRS101', 
                                    variable=models['ResNetRS101'], 
                                    onvalue=True, 
                                    offvalue=False)
model_ResNetRS101.grid(row=2, 
                       column=0, 
                       sticky=(tk.W + tk.E))

# ResNet 152
models["ResNet152"] = tk.BooleanVar()
model_ResNet152 = ttk.Checkbutton(mc_info, 
                                  text="ResNet152", 
                                  variable=models["ResNet152"], 
                                  onvalue=True, 
                                  offvalue=False)
model_ResNet152.grid(row=2, 
                     column=1, 
                     sticky=(tk.W + tk.E))

# ResNet 152 Version 2
models["ResNet152V2"] = tk.BooleanVar()
model_ResNet152V2 = ttk.Checkbutton(mc_info, 
                                    text="ResNet152V2", 
                                    variable=models["ResNet152V2"], 
                                    onvalue=True, 
                                    offvalue=False)
model_ResNet152V2.grid(row=2, 
                       column=2, 
                       sticky=(tk.W + tk.E))

# ResNetRS 152
models["ResNetRS152"] = tk.BooleanVar()
model_ResNetRS152 = ttk.Checkbutton(mc_info, 
                                    text='ResNetRS152', 
                                    variable=models['ResNetRS152'], 
                                    onvalue=True, 
                                    offvalue=False)
model_ResNetRS152.grid(row=2, 
                       column=3, 
                       sticky=(tk.W + tk.E))

# ResNetRS 200
models["ResNetRS200"] = tk.BooleanVar()
model_ResNetRS200 = ttk.Checkbutton(mc_info, 
                                    text='ResNetRS200', 
                                    variable=models['ResNetRS200'], 
                                    onvalue=True, 
                                    offvalue=False)
model_ResNetRS200.grid(row=2, 
                       column=4, 
                       sticky=(tk.W + tk.E))

# ResNetRS 270
models["ResNetRS270"] = tk.BooleanVar()
model_ResNetRS270 = ttk.Checkbutton(mc_info, 
                                    text='ResNetRS270', 
                                    variable=models['ResNetRS270'], 
                                    onvalue=True, 
                                    offvalue=False)
model_ResNetRS270.grid(row=2, 
                       column=5, 
                       sticky=(tk.W + tk.E))

# ResNetRS 350
models["ResNetRS350"] = tk.BooleanVar()
model_ResNetRS350 = ttk.Checkbutton(mc_info, 
                                    text='ResNetRS350', 
                                    variable=models['ResNetRS350'], 
                                    onvalue=True, 
                                    offvalue=False)
model_ResNetRS350.grid(row=2, 
                       column=6, 
                       sticky=(tk.W + tk.E))

# ResNetRS 420
models["ResNetRS420"] = tk.BooleanVar()
model_ResNetRS420 = ttk.Checkbutton(mc_info,
                                    text='ResNetRS420',
                                    variable=models['ResNetRS420'],
                                    onvalue=True, 
                                    offvalue=False)
model_ResNetRS420.grid(row=2,
                       column=7, 
                       sticky=(tk.W + tk.E))



# Inception V3
models["InceptionV3"] = tk.BooleanVar()
model_InceptionV3 = ttk.Checkbutton(mc_info, 
                                    text="InceptionV3", 
                                    variable=models["InceptionV3"], 
                                    onvalue=True, 
                                    offvalue=False)
model_InceptionV3.grid(row=3, 
                       column=0, 
                       sticky=(tk.W + tk.E))

# Incception ResNet Version 2
models["InceptionResNetV2"] = tk.BooleanVar()
model_InceptionResNetV2 = ttk.Checkbutton(mc_info, 
                                          text="InceptionResNetV2", 
                                          variable=models["InceptionResNetV2"], 
                                          onvalue=True, 
                                          offvalue=False)
model_InceptionResNetV2.grid(row=3, 
                             column=1, 
                             sticky=(tk.W + tk.E))

# MobileNet
models["MobileNet"] = tk.BooleanVar()
model_MobileNet = ttk.Checkbutton(mc_info, 
                                  text="MobileNet", 
                                  variable=models["MobileNet"], 
                                  onvalue=True, 
                                  offvalue=False)
model_MobileNet.grid(row=3, 
                     c010olumn=2, 
                     sticky=(tk.W + tk.E))

# MobileNet Version 2
models["MobileNetV2"] = tk.BooleanVar()
model_MobileNetV2 = ttk.Checkbutton(mc_info, 
                                    text="MobileNetV2", 
                                    variable=models["MobileNetV2"], 
                                    onvalue=True, 
                                    offvalue=False)
model_MobileNetV2.grid(row=3, 
                       column=3, 
                       sticky=(tk.W + tk.E))

# MobileNet Version 3 Small
models["MobileNetV3Small"] = tk.BooleanVar()
model_MobileNetV3Small = ttk.Checkbutton(mc_info, 
                                         text="MobileNetV3Small", 
                                         variable=models["MobileNetV3Small"], 
                                         onvalue=True, 
                                         offvalue=False)
model_MobileNetV3Small.grid(row=3, 
                            column=4, 
                            sticky=(tk.W + tk.E))

# MobileNet Version 3 Large
models["MobileNetV3Large"] = tk.BooleanVar()
model_MobileNetV3Large = ttk.Checkbutton(mc_info, 
                                         text="MobileNetV3Large", 
                                         variable=models["MobileNetV3Large"], 
                                         onvalue=True, 
                                         offvalue=False)
model_MobileNetV3Large.grid(row=3, 
                            column=5, 
                            sticky=(tk.W + tk.E))

# DenseNet 121
models["DenseNet121"] = tk.BooleanVar()
model_DenseNet121 = ttk.Checkbutton(mc_info, 
                                    text="DenseNet121", 
                                    variable=models["DenseNet121"], 
                                    onvalue=True, 
                                    offvalue=False)
model_DenseNet121.grid(row=3, 
                       column=6, 
                       sticky=(tk.W + tk.E))

# DenseNet 169
models["DenseNet169"] = tk.BooleanVar()
model_DenseNet169 = ttk.Checkbutton(mc_info, 
                                    text="DenseNet169", 
                                    variable=models["DenseNet169"], 
                                    onvalue=True, 
                                    offvalue=False)
model_DenseNet169.grid(row=3, 
                       column=7, 
                       sticky=(tk.W + tk.E))



# DenseNet 201
models["DenseNet201"] = tk.BooleanVar()
model_DenseNet201 = ttk.Checkbutton(mc_info, 
                                    text="DenseNet201", 
                                    variable=models["DenseNet201"], 
                                    onvalue=True, 
                                    offvalue=False)
model_DenseNet201.grid(row=4, 
                       column=0, 
                       sticky=(tk.W + tk.E))

# NASNet Mobile
models["NASNetMobile"] = tk.BooleanVar()
model_NASNetMobile = ttk.Checkbutton(mc_info, 
                                     text="NASNetMobile", 
                                     variable=models["NASNetMobile"], 
                                     onvalue=True, 
                                     offvalue=False)
model_NASNetMobile.grid(row=4, 
                        column=1, 
                        sticky=(tk.W + tk.E))

# NASNet Large
models["NASNetLarge"] = tk.BooleanVar()
model_NASNetLarge = ttk.Checkbutton(mc_info, text="NASNetLarge", variable=models["NASNetLarge"], onvalue = 1, offvalue = 0)
model_NASNetLarge.grid(row=4, column=2, sticky=(tk.W + tk.E))

# EfficientNet B0
models["EfficientNetB0"] = tk.BooleanVar()
model_EfficientNetB0 = ttk.Checkbutton(mc_info, text="EfficientNetB0", variable=models["EfficientNetB0"], onvalue = 1, offvalue = 0)
model_EfficientNetB0.grid(row=4, column=3, sticky=(tk.W + tk.E))

# EfficientNet B0 Version 2
models["EfficientNetB0V2"] = tk.BooleanVar()
model_EfficientNetB0V2 = ttk.Checkbutton(mc_info, text="EfficientNetB0V2", variable=models["EfficientNetB0V2"], onvalue = 1, offvalue = 0)
model_EfficientNetB0V2.grid(row=4, column=4, sticky=(tk.W + tk.E))

# EfficientNet B1
models["EfficientNetB1"] = tk.BooleanVar()
model_EfficientNetB1 = ttk.Checkbutton(mc_info, text="EfficientNetB1", variable=models["EfficientNetB1"], onvalue = 1, offvalue = 0)
model_EfficientNetB1.grid(row=4, column=5, sticky=(tk.W + tk.E))

# EfficientNet B1 Version 2
models["EfficientNetB1V2"] = tk.BooleanVar()
model_EfficientNetB1V2 = ttk.Checkbutton(mc_info, text="EfficientNetB1V2", variable=models["EfficientNetB1V2"], onvalue = 1, offvalue = 0)
model_EfficientNetB1V2.grid(row=4, column=6, sticky=(tk.W + tk.E))

# EfficientNet B2
models["EfficientNetB2"] = tk.BooleanVar()
model_EfficientNetB2 = ttk.Checkbutton(mc_info, text="EfficientNetB2", variable=models["EfficientNetB2"], onvalue = 1, offvalue = 0)
model_EfficientNetB2.grid(row=4, column=7, sticky=(tk.W + tk.E))



# EfficientNet B2 Version 2
models["EfficientNetB2V2"] = tk.BooleanVar()
model_EfficientNetB2V2 = ttk.Checkbutton(mc_info, text="EfficientNetB2V2", variable=models["EfficientNetB2V2"], onvalue = 1, offvalue = 0)
model_EfficientNetB2V2.grid(row=5, column=0, sticky=(tk.W + tk.E))

# EfficientNet B3
models["EfficientNetB3"] = tk.BooleanVar()
model_EfficientNetB3 = ttk.Checkbutton(mc_info, text="EfficientNetB3", variable=models["EfficientNetB3"], onvalue = 1, offvalue = 0)
model_EfficientNetB3.grid(row=5, column=1, sticky=(tk.W + tk.E))

# EfficientNet B3 Version 2
models["EfficientNetB3V2"] = tk.BooleanVar()
model_EfficientNetB3V2 = ttk.Checkbutton(mc_info, text="EfficientNetB3V2", variable=models["EfficientNetB3V2"], onvalue = 1, offvalue = 0)
model_EfficientNetB3V2.grid(row=5, column=2, sticky=(tk.W + tk.E))

# EfficientNet B4
models["EfficientNetB4"] = tk.BooleanVar()
model_EfficientNetB4 = ttk.Checkbutton(mc_info, text="EfficientNetB4", variable=models["EfficientNetB4"], onvalue = 1, offvalue = 0)
model_EfficientNetB4.grid(row=5, column=3, sticky=(tk.W + tk.E))

# EfficientNet B5
models["EfficientNetB5"] = tk.BooleanVar()
model_EfficientNetB5 = ttk.Checkbutton(mc_info, text="EfficientNetB5", variable=models["EfficientNetB5"], onvalue = 1, offvalue = 0)
model_EfficientNetB5.grid(row=5, column=4, sticky=(tk.W + tk.E))

# EfficientNet B6
models["EfficientNetB6"] = tk.BooleanVar()
model_EfficientNetB6 = ttk.Checkbutton(mc_info, text="EfficientNetB6", variable=models["EfficientNetB6"], onvalue = 1, offvalue = 0)
model_EfficientNetB6.grid(row=5, column=5, sticky=(tk.W + tk.E))

# EfficientNet B7
models["EfficientNetB7"] = tk.BooleanVar()
model_EfficientNetB7 = ttk.Checkbutton(mc_info, text="EfficientNetB7", variable=models["EfficientNetB7"], onvalue = 1, offvalue = 0)
model_EfficientNetB7.grid(row=5, column=6, sticky=(tk.W + tk.E))

# EfficientNet Version 2 Smalll
models["EfficientNetV2Small"] = tk.BooleanVar()
model_EfficientNetV2Small = ttk.Checkbutton(mc_info, text="EfficientNetV2Small", variable=models["EfficientNetV2Small"], onvalue = 1, offvalue = 0)
model_EfficientNetV2Small.grid(row=5, column=7, sticky=(tk.W + tk.E))



# EfficientNet Version 2 Medium
models["EfficientNetV2Medium"] = tk.BooleanVar()
model_EfficientNetV2Medium = ttk.Checkbutton(mc_info, text="EfficientNetV2Medium", variable=models["EfficientNetV2Medium"], onvalue = 1, offvalue = 0)
model_EfficientNetV2Medium.grid(row=6, column=0, sticky=(tk.W + tk.E))

# EfficientNet Version 2 Large
models["EfficientNetV2Large"] = tk.BooleanVar()
model_EfficientNetV2Large = ttk.Checkbutton(mc_info, text="EfficientNetV2Large", variable=models["EfficientNetV2Large"], onvalue = 1, offvalue = 0)
model_EfficientNetV2Large.grid(row=6, column=1, sticky=(tk.W + tk.E))

# ConvNetXt Tiny
models["ConvNeXtTiny"] = tk.BooleanVar()
model_ConvNeXtTiny = ttk.Checkbutton(mc_info, text="ConvNeXtTiny", variable=models["ConvNeXtTiny"], onvalue = 1, offvalue = 0)
model_ConvNeXtTiny.grid(row=6, column=2, sticky=(tk.W + tk.E))

# ConvNetXt Small
models["ConvNeXtSmall"] = tk.BooleanVar()
model_ConvNeXtSmall = ttk.Checkbutton(mc_info, text="ConvNeXtSmall", variable=models["ConvNeXtSmall"], onvalue = 1, offvalue = 0)
model_ConvNeXtSmall.grid(row=6, column=3, sticky=(tk.W + tk.E))

# ConvNetXt Base
models["ConvNeXtBase"] = tk.BooleanVar()
model_ConvNeXtBase = ttk.Checkbutton(mc_info, text="ConvNeXtBase", variable=models["ConvNeXtBase"], onvalue = 1, offvalue = 0)
model_ConvNeXtBase.grid(row=6, column=4, sticky=(tk.W + tk.E))

# ConvNetXt Large
models["ConvNeXtLarge"] = tk.BooleanVar()
model_ConvNeXtLarge = ttk.Checkbutton(mc_info, text="ConvNeXtLarge", variable=models["ConvNeXtLarge"], onvalue = 1, offvalue = 0)
model_ConvNeXtLarge.grid(row=6, column=5, sticky=(tk.W + tk.E))

# ConvNetXt XLarge
models["ConvNeXtXLarge"] = tk.BooleanVar()
model_ConvNeXtXLarge = ttk.Checkbutton(mc_info, text="ConvNeXtXLarge", variable=models["ConvNeXtXLarge"], onvalue = 1, offvalue = 0)
model_ConvNeXtXLarge.grid(row=6, column=6, sticky=(tk.W + tk.E))

# RegNetX002
models["RegNetX002"] = tk.BooleanVar()
model_RegNetX002 = ttk.Checkbutton(mc_info, text="RegNetX002", variable=models["RegNetX002"], onvalue = 1, offvalue = 0)
model_RegNetX002.grid(row=6, column=7, sticky=(tk.W + tk.E))



# RegNetY002
models["RegNetY002"] = tk.BooleanVar()
model_RegNetY002 = ttk.Checkbutton(mc_info, text="RegNetY002", variable=models["RegNetY002"], onvalue = 1, offvalue = 0)
model_RegNetY002.grid(row=7, column=0, sticky=(tk.W + tk.E))

# RegNetX004
models["RegNetX004"] = tk.BooleanVar()
model_RegNetX004 = ttk.Checkbutton(mc_info, text="RegNetX004", variable=models["RegNetX004"], onvalue = 1, offvalue = 0)
model_RegNetX004.grid(row=7, column=1, sticky=(tk.W + tk.E))

# RegNetY004
models["RegNetY004"] = tk.BooleanVar()
model_RegNetY004 = ttk.Checkbutton(mc_info, text="RegNetY004", variable=models["RegNetY004"], onvalue = 1, offvalue = 0)
model_RegNetY004.grid(row=7, column=2, sticky=(tk.W + tk.E))

# RegNetX006
models["RegNetX006"] = tk.BooleanVar()
model_RegNetX006 = ttk.Checkbutton(mc_info, text="RegNetX006", variable=models["RegNetX006"], onvalue = 1, offvalue = 0)
model_RegNetX006.grid(row=7, column=3, sticky=(tk.W + tk.E))

# RegNetY006
models["RegNetY006"] = tk.BooleanVar()
model_RegNetY006 = ttk.Checkbutton(mc_info, text="RegNetY006", variable=models["RegNetY006"], onvalue = 1, offvalue = 0)
model_RegNetY006.grid(row=7, column=4, sticky=(tk.W + tk.E))

# RegNetX008
models["RegNetX008"] = tk.BooleanVar()
model_RegNetX008 = ttk.Checkbutton(mc_info, text="RegNetX008", variable=models["RegNetX008"], onvalue = 1, offvalue = 0)
model_RegNetX008.grid(row=7, column=5, sticky=(tk.W + tk.E))

# RegNetY008
models["RegNetY008"] = tk.BooleanVar()
model_RegNetY008 = ttk.Checkbutton(mc_info, text="RegNetY008", variable=models["RegNetY008"], onvalue = 1, offvalue = 0)
model_RegNetY008.grid(row=7, column=6, sticky=(tk.W + tk.E))

# RegNetX016
models["RegNetX016"] = tk.BooleanVar()
model_RegNetX016 = ttk.Checkbutton(mc_info, text="RegNetX016", variable=models["RegNetX016"], onvalue = 1, offvalue = 0)
model_RegNetX016.grid(row=7, column=7, sticky=(tk.W + tk.E))



# RegNetY016
models["RegNetY016"] = tk.BooleanVar()
model_RegNetY016 = ttk.Checkbutton(mc_info, text="RegNetY016", variable=models["RegNetY016"], onvalue = 1, offvalue = 0)
model_RegNetY016.grid(row=8, column=0, sticky=(tk.W + tk.E))

# RegNetX032
models["RegNetX032"] = tk.BooleanVar()
model_RegNetX032 = ttk.Checkbutton(mc_info, text="RegNetX032", variable=models["RegNetX032"], onvalue = 1, offvalue = 0)
model_RegNetX032.grid(row=8, column=1, sticky=(tk.W + tk.E))

# RegNetY032
models["RegNetY032"] = tk.BooleanVar()
model_RegNetY032 = ttk.Checkbutton(mc_info, text="RegNetY032", variable=models["RegNetY032"], onvalue = 1, offvalue = 0)
model_RegNetY032.grid(row=8, column=2, sticky=(tk.W + tk.E))

# RegNetX040
models["RegNetX040"] = tk.BooleanVar()
model_RegNetX040 = ttk.Checkbutton(mc_info, text="RegNetX040", variable=models["RegNetX040"], onvalue = 1, offvalue = 0)
model_RegNetX040.grid(row=8, column=3, sticky=(tk.W + tk.E))

# RegNetY040
models["RegNetY040"] = tk.BooleanVar()
model_RegNetY040 = ttk.Checkbutton(mc_info, text="RegNetY040", variable=models["RegNetY040"], onvalue = 1, offvalue = 0)
model_RegNetY040.grid(row=8, column=4, sticky=(tk.W + tk.E))

# RegNetX064
models["RegNetX064"] = tk.BooleanVar()
model_RegNetX064 = ttk.Checkbutton(mc_info, text="RegNetX064", variable=models["RegNetX064"], onvalue = 1, offvalue = 0)
model_RegNetX064.grid(row=8, column=5, sticky=(tk.W + tk.E))

# RegNetY064
models["RegNetY064"] = tk.BooleanVar()
model_RegNetY064 = ttk.Checkbutton(mc_info, text="RegNetY064", variable=models["RegNetY064"], onvalue = 1, offvalue = 0)
model_RegNetY064.grid(row=8, column=6, sticky=(tk.W + tk.E))

# RegNetX080
models["RegNetX080"] = tk.BooleanVar()
model_RegNetX080 = ttk.Checkbutton(mc_info, text="RegNetX080", variable=models["RegNetX080"], onvalue = 1, offvalue = 0)
model_RegNetX080.grid(row=8, column=7, sticky=(tk.W + tk.E))



# RegNetY080
models["RegNetY080"] = tk.BooleanVar()
model_RegNetY080 = ttk.Checkbutton(mc_info, text="RegNetY080", variable=models["RegNetY080"], onvalue = 1, offvalue = 0)
model_RegNetY080.grid(row=9, column=0, sticky=(tk.W + tk.E))

# RegNetX120
models["RegNetX120"] = tk.BooleanVar()
model_RegNetX120 = ttk.Checkbutton(mc_info, text="RegNetX120", variable=models["RegNetX120"], onvalue = 1, offvalue = 0)
model_RegNetX120.grid(row=9, column=1, sticky=(tk.W + tk.E))

# RegNetY120
models["RegNetY120"] = tk.BooleanVar()
model_RegNetY120 = ttk.Checkbutton(mc_info, text="RegNetY120", variable=models["RegNetY120"], onvalue = 1, offvalue = 0)
model_RegNetY120.grid(row=9, column=2, sticky=(tk.W + tk.E))

# RegNetX160
models["RegNetX160"] = tk.BooleanVar()
model_RegNetX160 = ttk.Checkbutton(mc_info, text="RegNetX160", variable=models["RegNetX160"], onvalue = 1, offvalue = 0)
model_RegNetX160.grid(row=9, column=3, sticky=(tk.W + tk.E))

# RegNetY160
models["RegNetY160"] = tk.BooleanVar()
model_RegNetY160 = ttk.Checkbutton(mc_info, text="RegNetY160", variable=models["RegNetY160"], onvalue = 1, offvalue = 0)
model_RegNetY160.grid(row=9, column=4, sticky=(tk.W + tk.E))

# RegNetX320
models["RegNetX320"] = tk.BooleanVar()
model_RegNetX320 = ttk.Checkbutton(mc_info, text="RegNetX320", variable=models["RegNetX320"], onvalue = 1, offvalue = 0)
model_RegNetX320.grid(row=9, column=5, sticky=(tk.W + tk.E))

# RegNetY320
models["RegNetY320"] = tk.BooleanVar()
model_RegNetY320 = ttk.Checkbutton(mc_info, text="RegNetY320", variable=models["RegNetY320"], onvalue = 1, offvalue = 0)
model_RegNetY320.grid(row=9, column=6, sticky=(tk.W + tk.E))



def selectall():
    for i in models.keys():
        models[i].set(1)
        
def unselect():
    for i in models.keys():
        models[i].set(0)
    
# Select all
ttk.Button(mc_info, text="Select All", command=selectall).grid(row=10, column=6, padx=5, pady=5, sticky=(tk.W + tk.E))
ttk.Button(mc_info, text="Unselect All", command=unselect).grid(row=10, column=7, padx=5, pady=5, sticky=(tk.W + tk.E))

def sel():
    if (variables['strategie'].get()==1):
        optimizer2['state']='disabled'
        optimizer3['state']='disabled'
        
        loss2['state']='disabled'
        loss3['state']='disabled'
        
        epoch2['state']='disabled'
        epoch3['state']='disabled'
        
        lr2["state"] = 'disabled'
        lr3["state"] = 'disabled'

        lrdecay1['state']='readonly'
        lrdecay1.state(['selected'])
        lrdecay2['state']='disabled'
        lrdecay2.state(['!selected'])
        lrdecay3['state']='disabled'
        lrdecay3.state(['!selected'])

        earlystopping1['state']='readonly'
        earlystopping1.state(['selected'])
        earlystopping2['state']='disabled'
        earlystopping2.state(['!selected'])
        earlystopping3['state']='disabled'
        earlystopping3.state(['!selected'])
        
    elif(variables['strategie'].get()==2):
        optimizer2['state']='readonly'
        optimizer3['state']='disabled'
        
        loss2['state']='readonly'
        loss3['state']='disabled'
        
        epoch2['state']='readonly'
        epoch3['state']='disabled'
        
        lr2["state"] = 'readonly'
        lr3["state"] = 'disabled'
  
        lrdecay1['state']='disabled'
        lrdecay1.state(['!selected'])
        lrdecay2['state']='readonly'
        lrdecay2.state(['selected'])
        lrdecay3['state']='disabled'
        lrdecay3.state(['!selected'])      
  
        earlystopping1['state']='disabled'
        earlystopping1.state(['!selected'])
        earlystopping2['state']='readonly'
        earlystopping2.state(['selected'])
        earlystopping3['state']='disabled'
        earlystopping3.state(['!selected'])
        
    else:
        optimizer1['state']='readonly'
        optimizer2['state']='readonly'
        optimizer3['state']='readonly'
        
        loss1['state']='readonly'
        loss2['state']='readonly'
        loss3['state']='readonly'
        
        epoch1['state']='readonly'
        epoch2['state']='readonly'
        epoch3['state']='readonly'
        
        lr1["state"] = 'readonly'
        lr2["state"] = 'readonly'
        lr3["state"] = 'readonly'
 
        lrdecay1['state']='disabled'
        lrdecay1.state(['!selected'])
        lrdecay2['state']='disabled'
        lrdecay2.state(['!selected'])
        lrdecay3['state']='readonly'
        lrdecay3.state(['selected'])        
 
        earlystopping1['state']='disabled'
        earlystopping1.state(['!selected'])
        earlystopping2['state']='disabled'
        earlystopping2.state(['!selected'])
        earlystopping3['state']='readonly'
        earlystopping3.state(['selected'])       


train_info = ttk.LabelFrame(mc, text='Training parameters')
train_info.grid(padx=5, pady=5, sticky=(tk.W + tk.E))
for i in range(8):
    train_info.columnconfigure(i, weight=1)
    
variables['strategie'] = tk.IntVar()    
ttk.Label(train_info, text="Training step(s)").grid(row=0, 
                                                    column=0, 
                                                    padx=5, 
                                                    pady=5, 
                                                    sticky=(tk.W + tk.E))
ttk.Radiobutton(train_info, 
                text='1 Step', 
                value='1', 
                variable=variables['strategie'], 
                command=sel).grid(row=1, 
                                  column=0, 
                                  padx=5, 
                                  pady=5, 
                                  sticky=(tk.W + tk.E))
ttk.Radiobutton(train_info, 
                text='2 Steps', 
                value='2', 
                variable=variables['strategie'], 
                command=sel).grid(row=1, 
                                  column=1, 
                                  padx=5, 
                                  pady=5, 
                                  sticky=(tk.W + tk.E))
ttk.Radiobutton(train_info, 
                text='3 Steps', 
                value='3', 
                variable=variables['strategie'], 
                command=sel) .grid(row=1, 
                                   column=2, 
                                   padx=5, 
                                   pady=5, 
                                   sticky=(tk.W + tk.E))
variables['strategie'].set(1)

variables["multigpu"] = tk.IntVar()
ttk.Label(train_info, 
          text="Parallelization").grid(row=0, 
                                       column=3, 
                                       padx=5, 
                                       pady=5, 
                                       sticky=(tk.W + tk.E))
multi_gpu = ttk.Checkbutton(train_info, 
                            text="Multi GPU", 
                            variable=variables["multigpu"],
                            onvalue = 1, 
                            offvalue = 0)
multi_gpu.grid(row=1, 
               column=3, 
               padx=5, 
               pady=5, 
               sticky=(tk.W + tk.E))
multi_gpu.state(['disabled'])
variables["multigpu"].set(0)

if (numgpu > 1) :
    multi_gpu.state(['enabled'])
    multi_gpu.state(['selected'])
    variables["multigpu"].set(1)
    
variables['checkpoint'] = tk.IntVar()
ttk.Label(train_info, 
          text="Save & Restore").grid(row=0, 
                                      column=4, 
                                      padx=5, 
                                      pady=5, 
                                      sticky=(tk.W + tk.E))
chkpoint = ttk.Checkbutton(train_info, 
                           text="Checkpoint", 
                           variable=variables["checkpoint"])
chkpoint.grid(row=1, 
              column=4, 
              padx=5, 
              pady=5, 
              sticky=(tk.W + tk.E))
chkpoint.state(['selected'])
variables['checkpoint'].set(1)

variables['kerastuning'] = tk.IntVar()
ttk.Label(train_info, 
          text="Improvement").grid(row=0, 
                                   column=5, 
                                   padx=5, 
                                   pady=5, 
                                   sticky=(tk.W + tk.E))
kerastuning = ttk.Checkbutton(train_info, 
                              text="Keras Tuning", 
                              variable=variables["kerastuning"])
kerastuning.grid(row=1, 
                 column=5, 
                 padx=5, 
                 pady=5, 
                 sticky=(tk.W + tk.E))
kerastuning.state(['selected'])
variables['kerastuning'].set(0)
kerastuning['state']='disabled'

ttk.Separator(train_info, 
              orient='horizontal').grid(row=2, 
                                        columnspan=7, 
                                        padx=5, 
                                        pady=5, 
                                        sticky=(tk.W + tk.E))

listOptimizer = ('SGD',
                 'RMSProp',
                 'Adam',
                 'AdamW',
                 'Adadelta',
                 'Adagrad',
                 'Adamax',
                 'Adafactor',
                 'Nadam',
                 'Ftrl')

ttk.Label(train_info, text="Step 1").grid(row=4, 
                                          column=0, 
                                          padx=5, 
                                          pady=5, 
                                          sticky=(tk.W + tk.E))
ttk.Label(train_info, text="Step 2").grid(row=5, 
                                          column=0, 
                                          padx=5, 
                                          pady=5, 
                                          sticky=(tk.W + tk.E))
ttk.Label(train_info, text="Step 3").grid(row=6, 
                                          column=0, 
                                          padx=5, 
                                          pady=5, 
                                          sticky=(tk.W + tk.E))

variables['optimizer1'] = tk.StringVar()
variables['optimizer2'] = tk.StringVar()
variables['optimizer3'] = tk.StringVar()

ttk.Label(train_info, text="Optimizer").grid(row=3, 
                                             column=1, 
                                             padx=5, 
                                             pady=5, 
                                             sticky=(tk.W + tk.E))
optimizer1 = ttk.Combobox(train_info, 
                          values=listOptimizer, 
                          textvariable=variables['optimizer1'], 
                          state='readonly')
optimizer1.grid(row=4, 
                column=1, 
                padx=5, 
                pady=5, 
                sticky=(tk.W + tk.E))
optimizer1.current(2)
optimizer2 = ttk.Combobox(train_info, 
                          values=listOptimizer, 
                          textvariable=variables['optimizer2'], 
                          state='disabled')
optimizer2.grid(row=5,
                column=1, 
                padx=5, 
                pady=5, 
                sticky=(tk.W + tk.E))
optimizer2.current(2)
optimizer3 = ttk.Combobox(train_info, 
                          values=listOptimizer, 
                          textvariable=variables['optimizer3'], 
                          state='disabled')
optimizer3.grid(row=6, 
                column=1, 
                padx=5, 
                pady=5, 
                sticky=(tk.W + tk.E))
optimizer3.current(2)

listLoss = ('BinaryCrossentropy', 
            'CategoricalCrossentropy', 
            'SparseCategoricalCrossentropy', 
            'Poisson',
            'KLDivergence',
            'MeanSquaredError',
            'MeanAbsoluteError',
            'MeanAbsolutePercentageError',
            'MeanSquaredLogarithmicError',
            'CosineSimilarity',
            'Huber',
            'LogCosh',
            'Hinge',
            'SquaredHinge',
            'CategoricalHinge') 

variables['loss1'] = tk.StringVar()
variables['loss2'] = tk.StringVar()
variables['loss3'] = tk.StringVar()

ttk.Label(train_info, text="Loss").grid(row=3, 
                                        column=2, 
                                        padx=5, 
                                        pady=5, 
                                        sticky=(tk.W + tk.E))
loss1 = ttk.Combobox(train_info, 
                     values=listLoss, 
                     textvariable=variables['loss1'], 
                     state='reaonly')
loss1.grid(row=4, 
           column=2, 
           padx=5, 
           pady=5, 
           sticky=(tk.W + tk.E))
loss1.current(2)
loss2 = ttk.Combobox(train_info, 
                     values=listLoss, 
                     textvariable=variables['loss2'], 
                     state='disabled')
loss2.grid(row=5, 
           column=2, 
           padx=5, 
           pady=5, 
           sticky=(tk.W + tk.E))
loss2.current(2)
loss3 = ttk.Combobox(train_info, 
                     values=listLoss, 
                     textvariable=variables['loss3'], 
                     state='disabled')
loss3.grid(row=6, 
           column=2, 
           padx=5, 
           pady=5, 
           sticky=(tk.W + tk.E))
loss3.current(2)

listEpoch = list(range(1,501))

variables['epoch1'] = tk.StringVar()
variables['epoch2'] = tk.StringVar()
variables['epoch3'] = tk.StringVar()

ttk.Label(train_info, text="Epoh").grid(row=3, 
                                        column=3, 
                                        padx=5, 
                                        pady=5, 
                                        sticky=(tk.W + tk.E))
epoch1 = ttk.Combobox(train_info, 
                      values=listEpoch, 
                      textvariable=variables['epoch1'], 
                      state='readonly')
epoch1.grid(row=4, 
            column=3, 
            padx=5, 
            pady=5, 
            sticky=(tk.W + tk.E))
epoch1.current(9)
epoch2 = ttk.Combobox(train_info, 
                      values=listEpoch, 
                      textvariable=variables['epoch2'], 
                      state='disabled')
epoch2.grid(row=5, 
            column=3, 
            padx=5, 
            pady=5, 
            sticky=(tk.W + tk.E))
epoch2.current(9)
epoch3 = ttk.Combobox(train_info, 
                      values=listEpoch, 
                      textvariable=variables['epoch3'], 
                      state='disabled')
epoch3.grid(row=6, 
            column=3, 
            padx=5, 
            pady=5, 
            sticky=(tk.W + tk.E))
epoch3.current(49)

listlr = [0.1,0.01,0.001,0.0001,0.00001]

variables['lr1'] = tk.DoubleVar()
variables['lr2'] = tk.DoubleVar()
variables['lr3'] = tk.DoubleVar()

ttk.Label(train_info, text="Learning Rate").grid(row=3, 
                                                 column=4, 
                                                 padx=5, 
                                                 pady=5, 
                                                 sticky=(tk.W + tk.E))
lr1 = ttk.Combobox(train_info, 
                   values=listlr, 
                   textvariable=variables['lr1'], 
                   state='readonly')
lr1.grid(row=4, 
         column=4, 
         padx=5, 
         pady=5, 
         sticky=(tk.W + tk.E))
lr1.current(2)
lr2 = ttk.Combobox(train_info, 
                   values=listlr, 
                   textvariable=variables['lr2'], 
                   state='disabled')
lr2.grid(row=5, 
         column=4, 
         padx=5, 
         pady=5, 
         sticky=(tk.W + tk.E))
lr2.current(3)
lr3 = ttk.Combobox(train_info, 
                   values=listlr, 
                   textvariable=variables['lr3'], 
                   state='disabled')
lr3.grid(row=6, 
         column=4, 
         padx=5, 
         pady=5, 
         sticky=(tk.W + tk.E))
lr3.current(4)

variables['lrdecay1'] = tk.BooleanVar()
variables['lrdecay2'] = tk.BooleanVar()
variables['lrdecay3'] = tk.BooleanVar()

ttk.Label(train_info, 
          text="Lr Decay").grid(row=3, 
                                column=5, 
                                padx=5, 
                                pady=5, 
                                sticky=(tk.W + tk.E))
lrdecay1 = ttk.Checkbutton(train_info, 
                           text="", 
                           variable=variables["lrdecay1"])
lrdecay1.grid(row=4, 
              column=5, 
              padx=5, 
              pady=5)
lrdecay1.state(["selected"])
variables["lrdecay1"].set(1)

lrdecay2 = ttk.Checkbutton(train_info, 
                           text="", 
                           variable=variables["lrdecay2"])
lrdecay2.grid(row=5, 
              column=5, 
              padx=5, 
              pady=5)
lrdecay2.state(['!selected'])
lrdecay2['state']='disabled'
variables["lrdecay2"].set(0)

lrdecay3 = ttk.Checkbutton(train_info, 
                           text="", 
                           variable=variables["lrdecay3"])
lrdecay3.grid(row=6,
              column=5, 
              padx=5, 
              pady=5)
lrdecay3.state(['!selected'])
lrdecay3['state']='disabled'
variables["lrdecay3"].set(0)

variables['earlystopping1'] = tk.BooleanVar()
variables['earlystopping2'] = tk.BooleanVar()
variables['earlystopping3'] = tk.BooleanVar()

ttk.Label(train_info, 
          text="Early Stopping").grid(row=3, 
                                      column=6, 
                                      padx=5, 
                                      pady=5, 
                                      sticky=(tk.W + tk.E))
earlystopping1 = ttk.Checkbutton(train_info, 
                                 text="", 
                                 variable=variables['earlystopping1'])
earlystopping1.grid(row=4, 
                    column=6, 
                    padx=5, 
                    pady=5)
earlystopping1.state(["selected"])
variables["earlystopping1"].set(1)

earlystopping2 = ttk.Checkbutton(train_info, 
                                 text="", 
                                 variable=variables['earlystopping2'])
earlystopping2.grid(row=5, 
                    column=6, 
                    padx=5, 
                    pady=5)
earlystopping2.state(['!selected'])
earlystopping2['state']='disabled'
variables["earlystopping2"].set(0)

earlystopping3 = ttk.Checkbutton(train_info, 
                                 text="", 
                                 variable=variables['earlystopping3'])
earlystopping3.grid(row=6, 
                    column=6, 
                    padx=5, 
                    pady=5)
earlystopping3.state(['!selected'])
earlystopping3['state']='disabled'
variables["earlystopping3"].set(0)


# XAI / fINE Tuning
xai_info = ttk.LabelFrame(mc, text='Explainability')
xai_info.grid(padx=5, 
              pady=5, 
              sticky=(tk.W + tk.E))

for i in range(8):
    xai_info.columnconfigure(i, weight=1)
 
# ActivationMaximization
variables["activationmaximization"] = tk.IntVar()
activationmaximization = ttk.Checkbutton(xai_info, 
                                         text="Activation Maximization", 
                                         variable=variables["activationmaximization"], 
                                         onvalue = 1, 
                                         offvalue = 0)
activationmaximization.grid(row=0, 
                            column=0, 
                            sticky=(tk.W + tk.E))
activationmaximization.state(['selected'])
variables["activationmaximization"].set(0)
activationmaximization['state']='disabled'

variables["gradcam"] = tk.IntVar()
gradcam = ttk.Checkbutton(xai_info, 
                          text="GradCAM", 
                          variable=variables["gradcam"], 
                          onvalue = 1, 
                          offvalue = 0)
gradcam.grid(row=0, 
             column=1, 
             sticky=(tk.W + tk.E))
gradcam.state(['selected'])
variables["gradcam"].set(0)
gradcam['state']='disabled'

variables["gradcamplusplus"] = tk.IntVar()
gradcamplus = ttk.Checkbutton(xai_info, 
                              text="GradCAM++", 
                              variable=variables["gradcamplusplus"], 
                              onvalue = 1, 
                              offvalue = 0)
gradcamplus.grid(row=0, 
                 column=2, 
                 sticky=(tk.W + tk.E))
gradcamplus.state(['selected'])
variables["gradcamplusplus"].set(0)
gradcamplus['state']='disabled'

variables["scorecam"] = tk.IntVar()
scorecam = ttk.Checkbutton(xai_info, 
                           text="ScoreCAM", 
                           variable=variables["scorecam"], 
                           onvalue = 1, 
                           offvalue = 0)
scorecam.grid(row=0, 
              column=3, 
              sticky=(tk.W + tk.E))
scorecam.state(['selected'])
variables["scorecam"].set(0)
scorecam['state']='disabled'

variables["fasterscorecam"] = tk.IntVar()
fasterscorecam = ttk.Checkbutton(xai_info, 
                                 text="Faster-CAM", 
                                 variable=variables["fasterscorecam"], 
                                 onvalue = 1, 
                                 offvalue = 0)
fasterscorecam.grid(row=0, 
                    column=4, 
                    sticky=(tk.W + tk.E))
fasterscorecam.state(['selected'])
variables["fasterscorecam"].set(0)
fasterscorecam['state']='disabled'

variables["layercam"] = tk.IntVar()
layercam = ttk.Checkbutton(xai_info, 
                           text="LayerCAM", 
                           variable=variables["layercam"], 
                           onvalue = 1, 
                           offvalue = 0)
layercam.grid(row=0, 
              column=5, 
              sticky=(tk.W + tk.E))
layercam.state(['selected'])
variables["layercam"].set(0)
layercam['state']='disabled'

variables["vanillasaliency"] = tk.IntVar()
vanillasaliency = ttk.Checkbutton(xai_info, 
                                  text="Vanilla Saliency", 
                                  variable=variables["vanillasaliency"], 
                                  onvalue = 1, 
                                  offvalue = 0)
vanillasaliency.grid(row=0, 
                     column=6, 
                     sticky=(tk.W + tk.E))
vanillasaliency.state(['selected'])
variables["vanillasaliency"].set(0)
vanillasaliency['state']='disabled'

variables["smoothgrad"] = tk.IntVar()
smoothgrad = ttk.Checkbutton(xai_info, 
                             text="SmoothGrad", 
                             variable=variables["smoothgrad"], 
                             onvalue=1, 
                             offvalue=0)
smoothgrad.grid(row=0, 
                column=7, 
                sticky=(tk.W + tk.E))
smoothgrad.state(['selected'])
variables["smoothgrad"].set(0)
smoothgrad['state']='disabled'

# Output Section
output_info = ttk.LabelFrame(mc, text='Output')
output_info.grid(padx=5, 
                 pady=5, 
                 sticky=(tk.W + tk.E))

for i in range(5):
    output_info.columnconfigure(i, weight=1)
    
# Chechkbox Save Model
variables["savemodel"] = tk.IntVar()
savemodel = ttk.Checkbutton(output_info, 
                            text="Save Model", 
                            variable=variables['savemodel'], 
                            onvalue = 1, 
                            offvalue = 0)
savemodel.grid(row=0, 
               column=0, 
               sticky=(tk.W + tk.E))
savemodel.state(['selected'])
variables["savemodel"].set(1)

# Checkbox Training / Validation Graphs
variables["traingraph"] = tk.IntVar()
traingraph = ttk.Checkbutton(output_info, 
                             text="Training / Validation Graphs", 
                             variable=variables["traingraph"], 
                             onvalue = 1, 
                             offvalue = 0)
traingraph.grid(row=0, 
                column=1, 
                sticky=(tk.W + tk.E))
traingraph.state(['selected'])
variables["traingraph"].set(1)

# Confusion Matrix
variables["confmatrix"] = tk.IntVar()   
confmatrix = ttk.Checkbutton(output_info, 
                             text="Confusion Matrix", 
                             variable=variables["confmatrix"], 
                             onvalue = 1, 
                             offvalue = 0)
confmatrix.grid(row=0, 
                column=2, 
                sticky=(tk.W + tk.E))
confmatrix.state(['selected'])
variables["confmatrix"].set(1)

# Classification Report
variables["classreport"] = tk.IntVar()
classreport = ttk.Checkbutton(output_info, 
                              text="Classification Report", 
                              variable=variables["classreport"], 
                              onvalue = 1, 
                              offvalue = 0)
classreport.grid(row=0, 
                 column=3, 
                 sticky=(tk.W + tk.E))
classreport.state(['selected'])
variables["classreport"].set(1)

# Conversion of the model in TFLite
variables["tflite"] = tk.IntVar()
tflite = ttk.Checkbutton(output_info, 
                         text="TFLite Conversion", 
                         variable=variables["tflite"], 
                         onvalue = 1, 
                         offvalue = 0)
tflite.grid(row=0, 
            column=4, 
            sticky=(tk.W + tk.E))
tflite.state(['!selected'])



# Info Section
info_info = ttk.LabelFrame(mc, text='Info')
info_info.grid(padx=5, 
               pady=5, 
               sticky=(tk.W + tk.E))
for i in range(1):
    info_info.columnconfigure(i, weight=1)
    
ttk.Label(info_info, 
          text="GPUs Available: " 
          + str(numgpu) 
          + " - Python: " 
          + platform.python_version() 
          + " - TensorFlow: " 
          + tf.__version__ 
          + " - Keras: " 
          + k.__version__ 
          + " - Numpy: " 
          + np.version.version 
          + " - Pandas: " 
          + pd.__version__ 
          + " - Sklearn: " 
          + sk.__version__ 
          + " - Seaborn: " 
          + sns.__version__ 
          + "  - Matplotlib: " 
          + mpl.__version__).grid(row=0, column=0)



# Execution
exec_info = ttk.LabelFrame(mc, text='Execution')
exec_info.grid(padx=5, pady=5, sticky=(tk.W + tk.E))
for i in range(5):
    exec_info.columnconfigure(i, weight=1)

pgb = tk.IntVar()
pb = ttk.Progressbar(exec_info, 
                     orient='horizontal',
                     mode='determinate', 
                     variable=pgb).grid(row=0, 
                                        column=1, 
                                        columnspan=3, 
                                        sticky=(tk.E + tk.W))


def merge_dictionaries(dict1, dict2):
    #merged_dict = dict1.copy()
    #merged_dict.update(dict2)
    #return merged_dict
    res = dict1 | dict2
    return res 

def reset():
    print("reset")
    augment.state(['!selected'])
    variables["augment"].set(0)
    
    imgresize.current(1)
    channel.set(3)
    classes.set(4)
    valsplit.set(0.2)
    
    crop['state'] = 'disabled'
    crop.state(['!selected'])
    variables["crop"].set(0)
    horizflip['state'] = 'disabled'
    horizflip.state(['!selected'])
    variables["horizflip"].set(0)
    vertiflip['state'] = 'disabled'
    vertiflip.state(['!selected'])
    variables["vertiflip"].set(0)
    translation['state'] = 'disabled'
    translation.state(['selected'])
    variables["translation"].set(0)
    rotation['state'] = 'disabled'   
    rotation.state(['!selected'])
    variables["rotation"].set(0)
    zoom['state'] = 'disabled'
    zoom.state(['!selected'])
    variables["zoom"].set(0)
    contrast['state'] = 'disabled'
    contrast.state(['!selected'])
    variables["contrast"].set(0)
    brightness['state'] = 'disabled'
    brightness.state(['!selected'])
    variables["brightness"].set(0)
    
    unselect()  
      
    optimizer1.current(2)
    optimizer2.current(2)
    optimizer3.current(2)
    
    loss1.current(2)
    loss2.current(2)
    loss3.current(2)
    
    epoch1.current(9)
    epoch2.current(9)
    epoch3.current(49)
    
    lr1.current(2)
    lr2.current(3)
    lr3.current(4)
    
    savemodel.state(['selected'])
    variables["savemodel"].set(1)
    traingraph.state(['selected'])
    variables["traingraph"].set(1)
    confmatrix.state(['selected'])
    variables["confmatrix"].set(1)
    classreport.state(['selected'])
    variables["classreport"].set(1)
    tflite.state(['!selected'])
    variables["tflite"].set(0)   
    
def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
       return lr * tf.math.exp(-0.1)   
   
def pb_progress(cpt, total):
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()
        return cpt
    
def training(_img_height, 
             _img_width, 
             strategie, 
             multigpu, 
             base_model, 
             model_name, 
             _optimizer1, 
             _loss1, 
             _epoch1, 
             _lr1, 
             _optimizer2, 
             _loss2, 
             _epoch2, 
             _lr2, 
             _optimizer3, 
             _loss3, 
             _epoch3, 
             _lr3, 
             ds_train, 
             ds_valid, 
             savemodel, 
             traingraph, 
             confmatrix, 
             classreport, 
             tflite):
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        output_dir+'/model/'+model_name+".tf", 
        verbose=1, 
        save_best_only=True)
    
    model_earlystopping_callback =  tf.keras.callbacks.EarlyStopping(
        patience=5, 
        restore_best_weights=True)
    
    model_learningscheduler_callback = tf.keras.callbacks.LearningRateScheduler(
        scheduler, 
        verbose=1)
    
    callbacks = []
       
    if variables['checkpoint'].get() == 1:
        callbacks.append(model_checkpoint_callback)
        
    if strategie == 1:
        if variables["earlystopping1"].get() == 1:
            callbacks.append(model_earlystopping_callback)
            
        if variables["lrdecay1"].get() == 1:
            callbacks.append(model_learningscheduler_callback)
            
            
    if strategie == 2:
        if (variables["earlystopping2"].get() == 1):
            callbacks.append(model_earlystopping_callback)
            
        if (variables["lrdecay2"].get() == 1):
            callbacks.append(model_learningscheduler_callback)
            
            
    if strategie == 3:
        if (variables["earlystopping3"].get() == 1):
            callbacks.append(model_earlystopping_callback)
            
        if (variables["lrdecay3"].get() == 1):
            callbacks.append(model_learningscheduler_callback)
    
    print (model_name)
    
    if _loss1 == "SparseCategoricalCrossentropy":
        _loss1 = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        
    if _loss2 == "SparseCategoricalCrossentropy":
        _loss2 = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            
    if _loss3 == "SparseCategoricalCrossentropy":
         _loss3 = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)   
            
    
    # Multi GPU
    if multigpu == 1: 

        strategy = tf.distribute.MirroredStrategy()
        
        with strategy.scope():
           
            base_model.trainable = False
            
            # Add a new classifier layers on top of the base model
            inputs = tf.keras.Input(shape=(_img_height, 
                                           _img_width, 
                                           int(variables["channel"].get())))
            x = base_model(inputs, training=False)
            x = layers.GlobalAveragePooling2D()(x)
            outputs = layers.Dense(variables["classes"].get())(x) 
            model = tf.keras.Model(inputs, outputs)
            
            # Compile the model
            model.compile(optimizer=_optimizer1,
                          loss=_loss1,
                          metrics=['accuracy'])
        
        # Train the model
        hist = model.fit(ds_train, 
                         validation_data=ds_valid, 
                         epochs=int(_epoch1), 
                         callbacks=callbacks)   
        
        if strategie == 2:
            with strategy.scope():
                
                # Fine-tune the base model
                base_model.trainable = True
                
                model.compile(optimizer=_optimizer2,
                              loss=_loss2,
                              metrics=['accuracy'])
                
            hist2 = model.fit(ds_train, 
                              validation_data=ds_valid, 
                              epochs=int(_epoch2), 
                              callbacks=callbacks)    
                
        if strategie == 3:
            with strategy.scope():
                # Compile the model
                model.compile(optimizer=optimizer2,
                              loss=_loss2,
                              metrics=['accuracy'])
         
            # Train the model
            hist2 = model.fit(ds_train, 
                              validation_data=ds_valid, 
                              epochs=int(_epoch2), 
                              callbacks=callbacks)                
                
            
            with strategy.scope():
                
                # Fine-tune the base model
                base_model.trainable = True
                
                model.compile(optimizer=_optimizer3,
                              loss=_loss3,
                              metrics=['accuracy'])
                
            hist3 = model.fit(ds_train, 
                              validation_data=ds_valid, 
                              epochs=int(_epoch3), 
                              callbacks=callbacks)            
        
    # CPU or single GPU   
    else:
        base_model.trainable = False
        # Add a new classifier layers on top of the base model
        inputs = tf.keras.Input(shape=(_img_height, 
                                       _img_width, 
                                       int(variables["channel"].get())))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(variables["classes"].get())(x) 
        model = tf.keras.Model(inputs, outputs)
        
        # Compile the model
        model.compile(optimizer=_optimizer1,
                      loss=_loss1,
                      metrics=['accuracy'])
    
        # Train the model
        hist = model.fit(ds_train, 
                         validation_data=ds_valid, 
                         epochs=int(_epoch1), 
                         callbacks=callbacks) 

        if strategie == 2:
                
            # Fine-tune the base model
            base_model.trainable = True
            
            model.compile(optimizer=_optimizer2, 
                          loss=_loss2,
                          metrics=['accuracy'])
            
            hist2 = model.fit(ds_train, 
                              validation_data=ds_valid, 
                              epochs=int(_epoch2), 
                              callbacks=callbacks)    
                
        if strategie == 3:

            # Compile the model
            model.compile(optimizer=optimizer2,
                          loss=_loss2,
                          metrics=['accuracy'])
         
            # Train the model
            hist2 = model.fit(ds_train, 
                              validation_data=ds_valid, 
                              epochs=int(_epoch2), 
                              callbacks=callbacks)                
                
            # Fine-tune the base model
            base_model.trainable = True
            
            model.compile(optimizer=_optimizer3,
                          loss=_loss3,
                          metrics=['accuracy'])
                
            hist3 = model.fit(ds_train, validation_data=ds_valid, epochs=int(_epoch3), callbacks=callbacks)            

    
    #Output
    if savemodel == 1:
        model_json = model.to_json()
        with open(output_dir+'/model/'+model_name+'.json', 'w') as json_file:
            json_file.write(model_json)        

    if traingraph == 1:
        if (strategie == 1):
            hist_ = pd.DataFrame(hist.history)
        elif(strategie == 2):
            hist_ = pd.DataFrame(merge_dictionaries(hist.history, hist2.history))
        else:
            hist_ = pd.DataFrame(merge_dictionaries(merge_dictionaries(hist.history, hist2.history), hist3.history))
            
        hist_

        plt.figure(figsize=(15,5))
        plt.title(model_name)
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
        plt.savefig(output_dir+"/fig/"+model_name+'_model_performance.png')
        plt.show()
        
    if (confmatrix == 1):
        x_val,y_val,y_pred=[],[],[]
        for images, labels in ds_valid:
            y_val.extend(labels.numpy())
            x_val.extend(images.numpy())
        predictions=model.predict(np.array(x_val))
        for i in predictions:
            y_pred.append(np.argmax(i))
        df=pd.DataFrame()
        df['Actual'],df['Prediction']=y_val,y_pred
        df
        
        ax = plt.subplot()
        CM = confusion_matrix(y_val, y_pred)
        sns.heatmap(CM, annot=True, fmt='g', ax=ax, cbar=False, cmap='RdBu')
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels') 
        ax.set_title(model_name + ' - Confusion Matrix')
        plt.savefig(output_dir+"/fig/"+model_name+'_confusion_matrix.png')
        plt.show()
        CM
        
    if (classreport == 1):
        ClassificationReport = classification_report(y_val, y_pred)
        print('Classification Report is : ', ClassificationReport )
        
    if (tflite == 1):
        path = output_dir+'/model/'+model_name+".tf"    
            
        if os.path.exists(path) :
            print("Conversion in TFLite")
            # Convert the model.
            converter = tf.lite.TFLiteConverter.from_saved_model(path)
            tflite_model = converter.convert()
            
            # Save the model.
            with open(output_dir+'/model/'+model_name+'.tflite', 'wb') as f:
                f.write(tflite_model)
      
                
            # Convert the model.              
            converter2 = tf.lite.TFLiteConverter.from_saved_model(path)
            converter2.target_spec.supported_types = [tf.float16]
            tflite_model2 = converter2.convert()
            
            # Save the model.
            with open(output_dir+'/model/'+model_name+'-float16.tflite', 'wb') as f:
                f.write(tflite_model2)



def run():
    print("Start")
    
    size = variables["imgresizing"].get().split(' x ')
    img_height = int(size[0])
    img_width = int(size[1])
    
    # Load the data
    print("Data Processing...")
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
      data_dir,
      validation_split=float(variables["valsplit"].get()),
      subset="training",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=int(variables["batchsize"].get()))

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
      data_dir,
      validation_split=float(variables["valsplit"].get()),
      subset="validation",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=int(variables["batchsize"].get()))

    print("Dataset configuration")
    # Configure the dataset for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
     
    total = 0
    cpt = 0
    
    pgb.set(0)
    mc.update()

    # Progressbar configuration
    for i in models.keys():
        if (models[i].get() == 1):
            total = total + 1

 
    ### Xception Model ##    
    if (models["Xception"].get() == 1):
        model_name = "Xception"  
        
        base_model = Xception(input_shape=(img_height, 
                                           img_width, 
                                           int(variables["channel"].get())),
                              include_top=False,
                              weights='imagenet')
            
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(), 
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
    
        cpt = pb_progress(cpt, total)
        
    
    ### VGG16 Model ###
    if (models["VGG16"].get() == 1):
        model_name = "VGG16"
        
        base_model = VGG16(input_shape=(img_height, 
                                        img_width, 
                                        int(variables["channel"].get())),
                           include_top=False,
                           weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(), 
                 variables['loss2'].get(),
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
    
        cpt = pb_progress(cpt, total)
    
    
    ### VGG19 model ###
    if (models["VGG19"].get() == 1):
        model_name = "VGG19"
        
        base_model = VGG19(input_shape=(img_height, 
                                        img_width, 
                                        int(variables["channel"].get())),
                           include_top=False,
                           weights='imagenet')  
        
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(), 
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())

        cpt = pb_progress(cpt, total)
    
    
    ### ResNet50 ###
    if (models["ResNet50"].get() == 1):
        model_name = "ResNet50"
        
        base_model = ResNet50(input_shape=(img_height, img_width, int(variables["channel"].get())),
                              include_top=False,
                              weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(), 
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())

        cpt = pb_progress(cpt, total)
    

    ### ResNet50 V2 ###
    if (models["ResNet50V2"].get() == 1):
        model_name = "ResNet50_V2"
        
        base_model = ResNet50V2(input_shape=(img_height, 
                                             img_width, 
                                             int(variables["channel"].get())),
                                include_top=False,
                                weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(), 
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())

        cpt = pb_progress(cpt, total)
        
        
    ### ResNetRS50 ###
    if (models["ResNetRS50"].get() == 1):
        model_name = "ResNetRS50"
        
        base_model = ResNetRS50(input_shape=(img_height, 
                                             img_width, 
                                             int(variables["channel"].get())),
                              include_top=False,
                              weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(), 
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())

        cpt = pb_progress(cpt, total)
    
            
    ### ResNet101 ###
    if (models["ResNet101"].get() == 1):
        model_name = "ResNet101"
        
        base_model = ResNet101(input_shape=(img_height, 
                                            img_width, 
                                            int(variables["channel"].get())),
                               include_top=False,
                               weights='imagenet')
        
        
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(), 
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
     
        cpt = pb_progress(cpt, total)   
        
  
    ### ResNet101 V2 ###
    if (models["ResNet101V2"].get() == 1):
        model_name = "ResNet101_V2"
        
        base_model = ResNet101V2(input_shape=(img_height, 
                                              img_width, 
                                              int(variables["channel"].get())),
                                 include_top=False,
                                 weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(), 
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())

        cpt = pb_progress(cpt, total)
        
    
    ### ResNetRS101 ###
    if (models["ResNetRS101"].get() == 1):
        model_name = "ResNetRS101"
        
        base_model = ResNetRS101(input_shape=(img_height, 
                                              img_width, 
                                              int(variables["channel"].get())),
                              include_top=False,
                              weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(), 
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())

        cpt = pb_progress(cpt, total)
        
        
    ### ResNet152 ###
    if (models["ResNet152"].get() == 1):
        model_name = "ResNet152"
        base_model = ResNet152(input_shape=(img_height, 
                                            img_width, 
                                            int(variables["channel"].get())),
                               include_top=False,
                               weights='imagenet')
        
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(), 
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
    
        cpt = pb_progress(cpt, total)
        
        
    ### ResNet152 V2 ###
    if (models["ResNet152V2"].get() == 1):
        model_name = "ResNet152_V2"
        base_model = ResNet152V2(input_shape=(img_height, 
                                              img_width, 
                                              int(variables["channel"].get())),
                                 include_top=False,
                                 weights='imagenet')
        
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(), 
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
    
        cpt = pb_progress(cpt, total)  

 
    ### ResNetRS 152 ###
    if (models["ResNetRS152"].get() == 1):
        model_name = "ResNetRS152"
        base_model = ResNetRS152(input_shape=(img_height, 
                                              img_width, 
                                              int(variables["channel"].get())),
                              include_top=False,
                              weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(), 
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())

        cpt = pb_progress(cpt, total)


    ### ResNetRS 200 ###
    if (models["ResNetRS200"].get() == 1):
        model_name = "ResNetRS200"
        base_model = ResNetRS200(input_shape=(img_height, 
                                              img_width, 
                                              int(variables["channel"].get())),
                              include_top=False,
                              weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(), 
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())

        cpt = pb_progress(cpt, total)
  
    
    ### ResNetRS 270 ###
    if (models["ResNetRS270"].get() == 1):
        model_name = "ResNetRS270"
        base_model = ResNetRS270(input_shape=(img_height, 
                                              img_width, 
                                              int(variables["channel"].get())),
                              include_top=False,
                              weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(), 
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())

        cpt = pb_progress(cpt, total)
    
    
    ### ResNetRS 350 ###
    if (models["ResNetRS350"].get() == 1):
        model_name = "ResNetRS350"
        base_model = ResNetRS350(input_shape=(img_height, 
                                              img_width, 
                                              int(variables["channel"].get())),
                              include_top=False,
                              weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(), 
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(),
                 variables["classreport"].get(), 
                 variables["tflite"].get())

        cpt = pb_progress(cpt, total)  


    ### ResNetRS 420 ###
    if (models["ResNetRS420"].get() == 1):
        model_name = "ResNetRS420"
        base_model = ResNetRS420(input_shape=(img_height, 
                                              img_width, 
                                              int(variables["channel"].get())),
                              include_top=False,
                              weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(), 
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())

        cpt = pb_progress(cpt, total)
    
    
    ### Inception V3 ###
    if (models["InceptionV3"].get() == 1):
        model_name = "Inception_V3"
        base_model = InceptionV3(input_shape=(img_height, 
                                              img_width, 
                                              int(variables["channel"].get())),
                                 include_top=False,
                                 weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(), 
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
    
        cpt = pb_progress(cpt, total)  

    
    ### InceptionResNet V2 ###
    if (models["InceptionResNetV2"].get() == 1):
        model_name = "InceptionResNet_V2"
        base_model = InceptionResNetV2(input_shape=(img_height, 
                                                    img_width, 
                                                    int(variables["channel"].get())),
                                       include_top=False,
                                       weights='imagenet')
        
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(),
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
    
        cpt = pb_progress(cpt, total)
    
    
    ### MobileNet ###
    if (models["MobileNet"].get() == 1):
        model_name = "MobileNet"
        base_model = MobileNet(input_shape=(img_height, 
                                            img_width, 
                                            int(variables["channel"].get())),
                               include_top=False,
                               weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(),
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
    
        cpt = pb_progress(cpt, total)
    
    
    ### MobileNet V2 ###
    if (models["MobileNetV2"].get() == 1):
        model_name = "MobileNet_V2"
        base_model = MobileNetV2(input_shape=(img_height, 
                                              img_width, 
                                              int(variables["channel"].get())),
                                 include_top=False,
                                 weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(),
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
    
        cpt = pb_progress(cpt, total)
    
    
    ### MobileNet V3 Small ###
    if (models["MobileNetV3Small"].get() == 1):
        model_name = "MobileNet_V3_Small"
        base_model = MobileNetV3Small(input_shape=(img_height, 
                                                   img_width, 
                                                   int(variables["channel"].get())),
                                      include_top=False,
                                      weights='imagenet')
        
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(),
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
    
        cpt = pb_progress(cpt, total)
    
    
    ### MobileNet V3 Large ###
    if (models["MobileNetV3Large"].get() == 1):
        model_name = "MobileNet_V3_Large"

        base_model = MobileNetV3Large(input_shape=(img_height, 
                                                   img_width, 
                                                   int(variables["channel"].get())),
                                      include_top=False,
                                      weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(),
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())

        cpt = pb_progress(cpt, total)
    
    
    ### DenseNet 121 ###
    if (models["DenseNet121"].get() == 1):
        model_name = "DenseNet121"
        base_model = DenseNet121(input_shape=(img_height, 
                                              img_width, 
                                              int(variables["channel"].get())),
                                 include_top=False,
                                 weights='imagenet')
        
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(),
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
    
        cpt = pb_progress(cpt, total)
    
    
    ### DenseNet 169 ###
    if (models["DenseNet169"].get() == 1):
        model_name = "DenseNet169"
        base_model = DenseNet169(input_shape=(img_height, 
                                              img_width, 
                                              int(variables["channel"].get())),
                                 include_top=False,
                                 weights='imagenet')
        
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(),
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
    
        cpt = pb_progress(cpt, total)  
        
        
    ### DenseNet 201 ###
    if (models["DenseNet201"].get() == 1):
        model_name = "DenseNet201"
        base_model = DenseNet201(input_shape=(img_height, 
                                              img_width, 
                                              int(variables["channel"].get())),
                                 include_top=False,
                                 weights='imagenet')
        
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(),
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
    
        cpt = pb_progress(cpt, total)
    
    
    ### NASNetMobile ###
    if (models["NASNetMobile"].get() == 1):
        model_name = "NASNetMobile"
        base_model = NASNetMobile(input_shape=(img_height, 
                                               img_width, 
                                               int(variables["channel"].get())),
                                  include_top=False,
                                  weights='imagenet')
        
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(),
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
    
        cpt = pb_progress(cpt, total)
    
    
    ### NASNetLarge ###
    if (models["NASNetLarge"].get() == 1):
        model_name = "NASNetLarge"
        base_model = NASNetLarge(input_shape=(img_height, 
                                              img_width, 
                                              int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(),
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
    
        cpt = pb_progress(cpt, total)
    
    
    ### EfficientNetB0 ###
    if (models["EfficientNetB0"].get() == 1):
        model_name = "EfficientNet_B0"
        base_model = EfficientNetB0(input_shape=(img_height, 
                                                 img_width, 
                                                 int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(), 
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
    
        cpt = pb_progress(cpt, total)
    
    
    ### EfficientNetB0 V2 ###
    if (models["EfficientNetB0V2"].get() == 1):
        model_name = "EfficientNet_B0_V2"
        base_model = EfficientNetV2B0(input_shape=(img_height, 
                                                   img_width, 
                                                   int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(),
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
    
        cpt = pb_progress(cpt, total)


    ### EfficientNetB1 ###
    if (models["EfficientNetB1"].get() == 1):
        model_name = "EfficientNet_B1"
        base_model = EfficientNetB1(input_shape=(img_height, 
                                                 img_width, 
                                                 int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(),
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
    
        cpt = pb_progress(cpt, total)
    
  
    ### EfficientNetB1 V2 ###
    if (models["EfficientNetB1V2"].get() == 1):
        model_name = "EfficientNet_B1_V2"
        base_model = EfficientNetV2B1(input_shape=(img_height, 
                                                   img_width, 
                                                   int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(),
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
    
        cpt = pb_progress(cpt, total)
    
    
    ### EfficientNetB2 ###
    if (models["EfficientNetB2"].get() == 1):
        model_name = "EfficientNet_B2"
        base_model = EfficientNetB2(input_shape=(img_height, 
                                                 img_width, 
                                                 int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(),
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
    
        cpt = pb_progress(cpt, total)
    

    ### EfficientNetB2 V2 ###
    if (models["EfficientNetB2V2"].get() == 1):
        model_name = "EfficientNet_B2_V2"
        base_model = EfficientNetV2B2(input_shape=(img_height, 
                                                   img_width, 
                                                   int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(),
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
    
        cpt = pb_progress(cpt, total)
    
        
    ### EfficientNetB3 ###
    if (models["EfficientNetB3"].get() == 1):
        model_name = "EfficientNet_B3"
        base_model = EfficientNetB3(input_shape=(img_height, 
                                                 img_width, 
                                                 int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(),
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
    
        cpt = pb_progress(cpt, total)
    
  
    ### EfficientNetB3 V2 ###
    if (models["EfficientNetB3V2"].get() == 1):
        model_name = "EfficientNet_B3_V2"
        base_model = EfficientNetV2B3(input_shape=(img_height, 
                                                   img_width, 
                                                   int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(),
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
    
        cpt = pb_progress(cpt, total)
    

    ### EfficientNetB4 ###
    if (models["EfficientNetB4"].get() == 1):
        model_name = "EfficientNet_B4"
        base_model = EfficientNetB4(input_shape=(img_height, 
                                                 img_width, 
                                                 int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(),
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
    
        cpt = pb_progress(cpt, total)
    
    
    ### EfficientNetB5 ###
    if (models["EfficientNetB5"].get() == 1):
        model_name = "EfficientNet_B5"
        base_model = EfficientNetB5(input_shape=(img_height, 
                                                 img_width, 
                                                 int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(),
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
      
        cpt = pb_progress(cpt, total)
    
    
    ### EfficientNetB6 ###
    if (models["EfficientNetB6"].get() == 1):
        model_name = "EfficientNet_B6"
        base_model = EfficientNetB6(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(img_height, img_width, variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loss1'].get(), variables['epoch1'].get(), variables['lr1'].get(), variables['optimizer2'].get(), variables['loss2'].get(), variables['epoch2'].get(), variables['lr2'].get(), variables['optimizer3'].get(), variables['loss3'].get(), variables['epoch3'].get(), variables['lr3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = pb_progress(cpt, total)
    
    
    ### EfficientNetB7 ###
    if (models["EfficientNetB7"].get() == 1):
        model_name = "EfficientNet_B7"
        base_model = EfficientNetB7(input_shape=(img_height, 
                                                 img_width, 
                                                 int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
        
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(),
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
    
        cpt = pb_progress(cpt, total)
    
        
    ### EfficientNet2S ###
    if (models["EfficientNetV2Small"].get() == 1): 
        model_name = "EfficientNet_V2_Small"
        base_model = EfficientNetV2S(input_shape=(img_height, 
                                                  img_width, 
                                                  int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(),
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
    
        cpt = pb_progress(cpt, total)
    
    
    ### EfficientNet2M ###
    if (models["EfficientNetV2Medium"].get() == 1):
        model_name = "EfficientNet_V2_Medium"
        base_model = EfficientNetV2M(input_shape=(img_height, 
                                                  img_width, 
                                                  int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(),
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
    
        cpt = pb_progress(cpt, total)
    
    
    ### EfficientNet2L ###
    if (models["EfficientNetV2Large"].get() == 1):
        model_name = "EfficientNet_V2_Large"
        base_model = EfficientNetV2L(input_shape=(img_height, 
                                                  img_width, 
                                                  int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(),
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
    
        cpt = pb_progress(cpt, total)
    
    
    ### ConvNeXtTiny ###
    if (models["ConvNeXtTiny"].get() == 1):
        model_name = "ConvNeXtTiny"
        base_model = ConvNeXtTiny(input_shape=(img_height, 
                                               img_width, 
                                               int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(),
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
    
        cpt = pb_progress(cpt, total)
    
    
    ### ConvNeXtSmall ###
    if (models["ConvNeXtSmall"].get() == 1):
        model_name = "ConvNeXtSmall"
        base_model = ConvNeXtSmall(input_shape=(img_height,
                                                img_width, 
                                                int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(),
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
    
        cpt = pb_progress(cpt, total)
    
    
    ### ConvNeXtBase ###
    if (models["ConvNeXtBase"].get() == 1):
        model_name = "ConvNeXtBase"
        base_model = ConvNeXtBase(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(),
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
        
        cpt = pb_progress(cpt, total)
    
    
    ### ConvNeXtLarge ###
    if (models["ConvNeXtLarge"].get() == 1):
        model_name = "ConvNeXtLarge"
        base_model = ConvNeXtLarge(input_shape=(img_height, 
                                                img_width, 
                                                int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(),
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
    
        cpt = pb_progress(cpt, total)
    
    
    ### ConvNeXtXLarge ###
    if (models["ConvNeXtXLarge"].get() == 1):
        model_name = "ConvNeXtXLarge"
        base_model = ConvNeXtXLarge(input_shape=(img_height, 
                                                 img_width, 
                                                 int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(),
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
    
        cpt = pb_progress(cpt, total)
        
    ### RegNetX002 ###
    if (models["RegNetX002"].get() == 1):
        model_name = "RegNetX002"
        base_model = RegNetX002(input_shape=(img_height, 
                                             img_width, 
                                             int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(),
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
    
        cpt = pb_progress(cpt, total)  
    
    ### RegNetY002 ###
    if (models["RegNetY002"].get() == 1):
        model_name = "RegNetY002"
        base_model = RegNetY002(input_shape=(img_height, 
                                             img_width, 
                                             int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(),
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
    
        cpt = pb_progress(cpt, total) 
        
    ### RegNetX004 ###
    if (models["RegNetX004"].get() == 1):
        model_name = "RegNetX004"
        base_model = RegNetX004(input_shape=(img_height, 
                                             img_width, 
                                             int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(),
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
    
        cpt = pb_progress(cpt, total)  
    
    ### RegNetY004 ###
    if (models["RegNetY004"].get() == 1):
        model_name = "RegNetY004"
        base_model = RegNetY004(input_shape=(img_height, 
                                             img_width, 
                                             int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(),
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
    
        cpt = pb_progress(cpt, total)
        
    ### RegNetX006 ###
    if (models["RegNetX006"].get() == 1):
        model_name = "RegNetX006"
        base_model = RegNetX006(input_shape=(img_height, 
                                             img_width, 
                                             int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(),
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
    
        cpt = pb_progress(cpt, total)  
    
    ### RegNetY006 ###
    if (models["RegNetY006"].get() == 1):
        model_name = "RegNetY006"
        base_model = RegNetY006(input_shape=(img_height, 
                                             img_width, 
                                             int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(),
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
    
        cpt = pb_progress(cpt, total)         

    ### RegNetX008 ###
    if (models["RegNetX008"].get() == 1):
        model_name = "RegNetX008"
        base_model = RegNetX008(input_shape=(img_height, 
                                             img_width, 
                                             int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(),
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
    
        cpt = pb_progress(cpt, total)  
    
    ### RegNetY008 ###
    if (models["RegNetY008"].get() == 1):
        model_name = "RegNetY008"
        base_model = RegNetY008(input_shape=(img_height, 
                                             img_width, 
                                             int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(),
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
    
        cpt = pb_progress(cpt, total) 
        
    ### RegNetX016 ###
    if (models["RegNetX016"].get() == 1):
        model_name = "RegNetX016"
        base_model = RegNetX016(input_shape=(img_height, 
                                             img_width, 
                                             int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(),
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
    
        cpt = pb_progress(cpt, total)
    
    ### RegNetY016 ###
    if (models["RegNetY016"].get() == 1):
        model_name = "RegNetY016"
        base_model = RegNetY016(input_shape=(img_height, 
                                             img_width, 
                                             int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(),
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
    
        cpt = pb_progress(cpt, total)

    ### RegNetX032 ###
    if (models["RegNetX032"].get() == 1):
        model_name = "RegNetX032"
        base_model = RegNetX032(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(),
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
   
        cpt = pb_progress(cpt, total)  
    
    ### RegNetY032 ###
    if models["RegNetY032"].get() == 1:
        model_name = "RegNetY032"
        base_model = RegNetY032(input_shape=(img_height, 
                                             img_width, 
                                             int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(),
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
    
        cpt = pb_progress(cpt, total)

    ### RegNetX040 ###
    if models["RegNetX040"].get() == 1:
        model_name = "RegNetX040"
        base_model = RegNetX040(input_shape=(img_height, 
                                             img_width, 
                                             int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(),
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
    
        cpt = pb_progress(cpt, total)   
    
    ### RegNetY040 ###
    if models["RegNetY040"].get() == 1:
        model_name = "RegNetY040"
        base_model = RegNetY040(input_shape=(img_height, 
                                             img_width, 
                                             int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(),
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
   
        cpt = pb_progress(cpt, total)

    ### RegNetX064 ###
    if models["RegNetX064"].get() == 1:
        model_name = "RegNetX064"
        base_model = RegNetX064(input_shape=(img_height, 
                                             img_width, 
                                             int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(),
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
   
        cpt = pb_progress(cpt, total)   
    
    ### RegNetY064 ###
    if models["RegNetY064"].get() == 1:
        model_name = "RegNetY064"
        base_model = RegNetY064(input_shape=(img_height, 
                                             img_width, 
                                             int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(),
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
    
        cpt = pb_progress(cpt, total)
        
    ### RegNetX080 ###
    if models["RegNetX080"].get() == 1:
        model_name = "RegNetX080"
        base_model = RegNetX080(input_shape=(img_height, 
                                             img_width, 
                                             int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(),
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
    
        cpt = pb_progress(cpt, total)   
    
    ### RegNetY080 ###
    if (models["RegNetY080"].get() == 1):
        model_name = "RegNetY080"
        base_model = RegNetY080(input_shape=(img_height, 
                                             img_width, 
                                             int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(),
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
   
        cpt = pb_progress(cpt, total)  
        
    ### RegNetX120 ###
    if (models["RegNetX120"].get() == 1):
        model_name = "RegNetX120"
        base_model = RegNetX120(input_shape=(img_height, 
                                             img_width, 
                                             int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(),
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
    
        cpt = pb_progress(cpt, total)  
    
    ### RegNetY120 ###
    if (models["RegNetY120"].get() == 1):
        model_name = "RegNetY120"
        base_model = RegNetY120(input_shape=(img_height, 
                                             img_width, 
                                             int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(),
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
    
        cpt = pb_progress(cpt, total)
        
    ### RegNetX160 ###
    if (models["RegNetX160"].get() == 1):
        model_name = "RegNetX160"
        base_model = RegNetX160(input_shape=(img_height, 
                                             img_width, 
                                             int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(),
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
    
        cpt = pb_progress(cpt, total)   
    
    ### RegNetY160 ###
    if (models["RegNetY160"].get() == 1):
        model_name = "RegNetY160"
        base_model = RegNetY160(input_shape=(img_height, 
                                             img_width, 
                                             int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(),
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
    
        cpt = pb_progress(cpt, total)
 
    ### RegNetX320 ###
    if (models["RegNetX320"].get() == 1):
        model_name = "RegNetX320"
        base_model = RegNetX320(input_shape=(img_height, 
                                             img_width, 
                                             int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(),
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
    
        cpt = pb_progress(cpt, total)   
    
    ### RegNetY080 ###
    if (models["RegNetY320"].get() == 1):
        model_name = "RegNetY320"
        base_model = RegNetY320(input_shape=(img_height, 
                                             img_width, 
                                             int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(img_height, 
                 img_width, 
                 variables['strategie'].get(), 
                 variables["multigpu"].get(), 
                 base_model, 
                 model_name, 
                 variables['optimizer1'].get(), 
                 variables['loss1'].get(), 
                 variables['epoch1'].get(), 
                 variables['lr1'].get(), 
                 variables['optimizer2'].get(),
                 variables['loss2'].get(), 
                 variables['epoch2'].get(), 
                 variables['lr2'].get(), 
                 variables['optimizer3'].get(), 
                 variables['loss3'].get(), 
                 variables['epoch3'].get(), 
                 variables['lr3'].get(), 
                 train_ds, 
                 val_ds, 
                 variables["savemodel"].get(), 
                 variables["traingraph"].get(), 
                 variables["confmatrix"].get(), 
                 variables["classreport"].get(), 
                 variables["tflite"].get())
    
        cpt = pb_progress(cpt, total)
        
    print ("End")

# Execution 
ttk.Button(exec_info, text="Reset", command=reset).grid(row=0, 
                                                        column=0, 
                                                        padx=5, 
                                                        pady=5, 
                                                        sticky=(tk.W + tk.E))
ttk.Button(exec_info, text="Run", command=run).grid(row=0, 
                                                    column=4, 
                                                    padx=5, 
                                                    pady=5, 
                                                    sticky=(tk.W + tk.E))


# Show the window 
root.mainloop()