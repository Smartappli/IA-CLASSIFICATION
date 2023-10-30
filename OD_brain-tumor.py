# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 18:06:12 2023

@author: UMONS - 532807
"""

import tkinter as tk
from tkinter import ttk
import tensorflow as tf
import keras  as k
from tensorflow.keras import layers
from tensorflow.keras.applications import Xception, VGG16, VGG19
from tensorflow.keras.applications import ResNet50, ResNet101, ResNet152
from tensorflow.keras.applications import ResNet50V2, ResNet101V2, ResNet152V2
from tensorflow.keras.applications import RegNetX002, RegNetX004, RegNetX006, RegNetX008, RegNetX016, RegNetX032, RegNetX040, RegNetX064, RegNetX080, RegNetX120, RegNetX160, RegNetX320
from tensorflow.keras.applications import RegNetY002, RegNetY004, RegNetY006, RegNetY008, RegNetY016, RegNetY032, RegNetY040, RegNetY064, RegNetY080, RegNetY120, RegNetY160, RegNetY320
from tensorflow.keras.applications import ResNetRS50, ResNetRS101, ResNetRS152, ResNetRS200, ResNetRS270, ResNetRS350, ResNetRS420
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import MobileNet, MobileNetV2, MobileNetV3Small, MobileNetV3Large
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import NASNetMobile, NASNetLarge
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from tensorflow.keras.applications import EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2, EfficientNetV2B3
from tensorflow.keras.applications import EfficientNetV2S, EfficientNetV2M, EfficientNetV2L
from tensorflow.keras.applications import ConvNeXtTiny, ConvNeXtSmall, ConvNeXtBase, ConvNeXtLarge, ConvNeXtXLarge
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from sklearn.metrics import confusion_matrix, classification_report
import sklearn as sk
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import os.path
from pathlib import Path
import platform

numgpu = len(tf.config.list_physical_devices('GPU'))
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

variables = dict()
 
p = Path()
if (platform.system() == "Windows"):
    data_dir = str(p) + '\Data'
    output_dir = str(p) + '\Output'  
else :
    data_dir = str(p) + '/Data'
    output_dir = str(p) + '/Output'

batch_size = 32
img_height = 224 #256
img_width = 224 #256

root = tk.Tk()
root.title('AI Classifier')
root.columnconfigure(0, weight=1)

mc = ttk.Frame(root)
mc.grid(padx=10, pady=10, sticky=(tk.W + tk.E))
mc.columnconfigure(0, weight=1)



data_info = ttk.LabelFrame(mc, text='Data Parameters')
data_info.grid(padx=5, pady=5, sticky=(tk.W + tk.E))
for i in range(4):
    data_info.columnconfigure(i, weight=1)
 
variables["datapath"] = tk.StringVar()
ttk.Label(data_info, text="Data Path").grid(row=0, column=0, sticky=(tk.W + tk.E), padx=5, pady=5)
datapath = ttk.Entry(data_info, textvariable=variables["datapath"])
datapath.grid(row=1, columnspan=2, padx=5, pady=5, sticky=(tk.W + tk.E))
variables["datapath"].set(data_dir)

variables["outputdata"] = tk.StringVar()
ttk.Label(data_info, text="Output Path").grid(row=0, column=3, sticky=(tk.W + tk.E), padx=5, pady=5)
outputpath = ttk.Entry(data_info, textvariable=variables["outputdata"])
outputpath.grid(row=1, columnspan=2, column=3, padx=5, pady=5, sticky=(tk.W + tk.E))
variables["outputdata"].set(output_dir)

variables["imgresizing"] = tk.StringVar()
ttk.Label(data_info, text="Image Resizing (pixels)").grid(row=2, column=0, sticky=(tk.W + tk.E), padx=5, pady=5)
listsize = ['128 x 128', '224 x 224', '256 x 256', '300 x 300', '400 x 400', '512 x 512']
imgresize = ttk.Combobox(data_info, values=listsize, textvariable=variables["imgresizing"], state='readonly')
imgresize.grid(row=3, column=0, sticky=(tk.W + tk.E), padx=5, pady=5)
imgresize.current(1)

variables["channel"] = tk.IntVar()
ttk.Label(data_info, text="Number of channels").grid(row=2, column=1, sticky=(tk.W + tk.E), padx=5, pady=5)
channel = ttk.Spinbox(data_info, textvariable=variables["channel"], from_=1, to=4, increment=1, state='readonly')
channel.grid(row=3, column=1, sticky=(tk.W + tk.E), padx=5, pady=5)
channel.set(3)

variables["classes"] = tk.IntVar()
ttk.Label(data_info, text="Number of classes").grid(row=2, column=2, sticky=(tk.W + tk.E), padx=5, pady=5)
classes = ttk.Spinbox(data_info, textvariable=variables["classes"], from_=2, to=1000, increment=1, state='readonly')
classes.grid(row=3, column=2, sticky=(tk.W + tk.E), padx=5, pady=5)
classes.set(4)

variables["valsplit"] = tk.DoubleVar()
ttk.Label(data_info, text="Validation Split").grid(row=2, column=3, sticky=(tk.W + tk.E), padx=5, pady=5)
valsplit = ttk.Spinbox(data_info, textvariable=variables["valsplit"], from_=0, to=1, increment=0.01, state='readonly')
valsplit.grid(row=3, column=3, sticky=(tk.W + tk.E), padx=5, pady=5)
valsplit.set(0.2)

variables["batchsize"] = tk.IntVar()
ttk.Label(data_info, text="Batch Size").grid(row=2, column=4, sticky=(tk.W + tk.E), padx=5, pady=5)
listbatch = [1,2,4,8,16,32,64,128,256,512]
batchsize = ttk.Combobox(data_info, values=listbatch, textvariable=variables["batchsize"], state='readonly')
batchsize.grid(row=3, column=4, sticky=(tk.W + tk.E), padx=5, pady=5)
batchsize.current(5)



augment_info = ttk.LabelFrame(mc, text='Data Augmentation')
augment_info.grid(padx=5, pady=5, sticky=(tk.W + tk.E))
for i in range(8):
    augment_info.columnconfigure(i, weight=1)
   

   
def augment_sel():
    if (variables["augment"].get() == 0):
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
augment = ttk.Checkbutton(augment_info, text="Activate Data Augmentation", variable=variables["augment"], command=augment_sel)
augment.grid(row=0, columnspan=8)

ttk.Separator(augment_info, orient='horizontal').grid(row=1, columnspan=8, padx=5, pady=5, sticky=(tk.W + tk.E))

variables["crop"] = tk.IntVar() 
crop = ttk.Checkbutton(augment_info, text="Cropping", variable=variables["crop"], onvalue = 1, offvalue = 0)
crop.grid(row=2, column=0, sticky=(tk.W + tk.E), padx=5, pady=5)
crop['state'] = 'disabled'

variables['horizflip'] = tk.IntVar()
horizflip = ttk.Checkbutton(augment_info, text="Horizontal Flip", variable=variables["horizflip"], onvalue = 1, offvalue = 0)
horizflip.grid(row=2, column=1, sticky=(tk.W + tk.E), padx=5, pady=5)
horizflip['state'] = 'disabled'

variables['vertiflip'] = tk.IntVar()
vertiflip = ttk.Checkbutton(augment_info, text="Verticial Flip", variable=variables["vertiflip"], onvalue = 1, offvalue = 0)
vertiflip.grid(row=2, column=2, sticky=(tk.W + tk.E), padx=5, pady=5)
vertiflip['state'] = 'disabled'

variables['translation'] = tk.IntVar()
translation = ttk.Checkbutton(augment_info, text="Translation", variable=variables["translation"], onvalue = 1, offvalue = 0)
translation.grid(row=2, column=3, sticky=(tk.W + tk.E), padx=5, pady=5)
translation['state'] = 'disabled'

variables['rotation'] = tk.IntVar()
rotation = ttk.Checkbutton(augment_info, text="Rotation", variable=variables["rotation"], onvalue = 1, offvalue = 0)
rotation.grid(row=2, column=4, sticky=(tk.W + tk.E), padx=5, pady=5)
rotation['state'] = 'disabled'

variables['zoom'] = tk.IntVar()
zoom = ttk.Checkbutton(augment_info, text="Zoom", variable=variables["zoom"], onvalue = 1, offvalue = 0)
zoom.grid(row=2, column=5, sticky=(tk.W + tk.E), padx=5, pady=5)
zoom['state'] = 'disabled'

variables['contrast'] = tk.IntVar()
contrast = ttk.Checkbutton(augment_info, text="Contrast", variable=variables["contrast"], onvalue = 1, offvalue = 0)
contrast.grid(row=2, column=6, sticky=(tk.W + tk.E), padx=5, pady=5)
contrast['state'] = 'disabled'

variables['brightness'] = tk.IntVar()
brightness = ttk.Checkbutton(augment_info, text="Brightness", variable=variables["brightness"], onvalue = 1, offvalue = 0)
brightness.grid(row=2, column=7, sticky=(tk.W + tk.E), padx=5, pady=5)
brightness['state'] = 'disabled'



mc_info = ttk.LabelFrame(mc, text='Model(s) selection')
mc_info.grid(padx=5, pady=5, sticky=(tk.W + tk.E))
for i in range(8):
    mc_info.columnconfigure(i, weight=1)
    
# Xception       
variables['Xception'] = tk.BooleanVar()
model_Xception = ttk.Checkbutton(mc_info, text="Xception", variable=variables['Xception'], onvalue = 1, offvalue = 0)
model_Xception.grid(row=1, column=0, sticky=(tk.W + tk.E))

# VGG 16
variables['VGG16'] = tk.BooleanVar()
model_VGG16 = ttk.Checkbutton(mc_info, text="VGG16", variable=variables['VGG16'], onvalue = 1, offvalue = 0)
model_VGG16.grid(row=1, column=1, sticky=(tk.W + tk.E))

# VGG 19
variables['VGG19'] = tk.BooleanVar()
model_VGG19 = ttk.Checkbutton(mc_info, text="VGG19", variable=variables['VGG19'], onvalue = 1, offvalue = 0)
model_VGG19.grid(row=1, column=2, sticky=(tk.W + tk.E))

# ResNet 50
variables['ResNet50'] = tk.BooleanVar()
model_ResNet50 = ttk.Checkbutton(mc_info, text="ResNet50", variable=variables['ResNet50'], onvalue = 1, offvalue = 0)
model_ResNet50.grid(row=1, column=3, sticky=(tk.W + tk.E))

# ResNet 50 Version 2
variables['ResNet50V2'] = tk.BooleanVar()
model_ResNet50V2 = ttk.Checkbutton(mc_info, text="ResNet50V2", variable=variables['ResNet50V2'], onvalue = 1, offvalue = 0)
model_ResNet50V2.grid(row=1, column=4, sticky=(tk.W + tk.E))

# ResNetRS 50 
variables["ResNetRS50"] = tk.BooleanVar()
model_ResNetRS50 = ttk.Checkbutton(mc_info, text='ResNetRS50', variable=variables['ResNetRS50'], onvalue=1, offvalue=0)
model_ResNetRS50.grid(row=1, column=5, sticky=(tk.W + tk.E))

# ResNet 101
variables['ResNet101'] = tk.BooleanVar()
model_ResNet101 = ttk.Checkbutton(mc_info, text="ResNet101", variable=variables["ResNet101"], onvalue = 1, offvalue = 0)
model_ResNet101.grid(row=1, column=6, sticky=(tk.W + tk.E))

# ResNet 101 Version 2
variables["ResNet101V2"] = tk.BooleanVar()
model_ResNet101V2 = ttk.Checkbutton(mc_info, text="ResNet101V2", variable=variables["ResNet101V2"], onvalue = 1, offvalue = 0)
model_ResNet101V2.grid(row=1, column=7, sticky=(tk.W + tk.E))



# ResNetRS 101
variables["ResNetRS101"] = tk.BooleanVar()
model_ResNetRS101 = ttk.Checkbutton(mc_info, text='ResNetRS101', variable=variables['ResNetRS101'], onvalue=1, offvalue=0)
model_ResNetRS101.grid(row=2, column=0, sticky=(tk.W + tk.E))

# ResNet 152
variables["ResNet152"] = tk.BooleanVar()
model_ResNet152 = ttk.Checkbutton(mc_info, text="ResNet152", variable=variables["ResNet152"], onvalue = 1, offvalue = 0)
model_ResNet152.grid(row=2, column=1, sticky=(tk.W + tk.E))

# ResNet 152 Version 2
variables["ResNet152V2"] = tk.BooleanVar()
model_ResNet152V2 = ttk.Checkbutton(mc_info, text="ResNet152V2", variable=variables["ResNet152V2"], onvalue = 1, offvalue = 0)
model_ResNet152V2.grid(row=2, column=2, sticky=(tk.W + tk.E))

# ResNetRS 152
variables["ResNetRS152"] = tk.BooleanVar()
model_ResNetRS152 = ttk.Checkbutton(mc_info, text='ResNetRS152', variable=variables['ResNetRS152'], onvalue=1, offvalue=0)
model_ResNetRS152.grid(row=2, column=3, sticky=(tk.W + tk.E))

# ResNetRS 200
variables["ResNetRS200"] = tk.BooleanVar()
model_ResNetRS200 = ttk.Checkbutton(mc_info, text='ResNetRS200', variable=variables['ResNetRS200'], onvalue=1, offvalue=0)
model_ResNetRS200.grid(row=2, column=4, sticky=(tk.W + tk.E))

# ResNetRS 270
variables["ResNetRS270"] = tk.BooleanVar()
model_ResNetRS270 = ttk.Checkbutton(mc_info, text='ResNetRS270', variable=variables['ResNetRS270'], onvalue=1, offvalue=0)
model_ResNetRS270.grid(row=2, column=5, sticky=(tk.W + tk.E))

# ResNetRS 350
variables["ResNetRS350"] = tk.BooleanVar()
model_ResNetRS350 = ttk.Checkbutton(mc_info, text='ResNetRS350', variable=variables['ResNetRS350'], onvalue=1, offvalue=0)
model_ResNetRS350.grid(row=2, column=6, sticky=(tk.W + tk.E))

# ResNetRS 420
variables["ResNetRS420"] = tk.BooleanVar()
model_ResNetRS420 = ttk.Checkbutton(mc_info, text='ResNetRS420', variable=variables['ResNetRS420'], onvalue=1, offvalue=0)
model_ResNetRS420.grid(row=2, column=7, sticky=(tk.W + tk.E))



# Inception V3
variables["InceptionV3"] = tk.BooleanVar()
model_InceptionV3 = ttk.Checkbutton(mc_info, text="InceptionV3", variable=variables["InceptionV3"], onvalue = 1, offvalue = 0)
model_InceptionV3.grid(row=3, column=0, sticky=(tk.W + tk.E))

# Incception ResNet Version 2
variables["InceptionResNetV2"] = tk.BooleanVar()
model_InceptionResNetV2 = ttk.Checkbutton(mc_info, text="InceptionResNetV2", variable=variables["InceptionResNetV2"], onvalue = 1, offvalue = 0)
model_InceptionResNetV2.grid(row=3, column=1, sticky=(tk.W + tk.E))

# MobileNet
variables["MobileNet"] = tk.BooleanVar()
model_MobileNet = ttk.Checkbutton(mc_info, text="MobileNet", variable=variables["MobileNet"], onvalue = 1, offvalue = 0)
model_MobileNet.grid(row=3, column=2, sticky=(tk.W + tk.E))

# MobileNet Version 2
variables["MobileNetV2"] = tk.BooleanVar()
model_MobileNetV2 = ttk.Checkbutton(mc_info, text="MobileNetV2", variable=variables["MobileNetV2"], onvalue = 1, offvalue = 0)
model_MobileNetV2.grid(row=3, column=3, sticky=(tk.W + tk.E))

# MobileNet Version 3 Small
variables["MobileNetV3Small"] = tk.BooleanVar()
model_MobileNetV3Small = ttk.Checkbutton(mc_info, text="MobileNetV3Small", variable=variables["MobileNetV3Small"], onvalue = 1, offvalue = 0)
model_MobileNetV3Small.grid(row=3, column=4, sticky=(tk.W + tk.E))

# MobileNet Version 3 Large
variables["MobileNetV3Large"] = tk.BooleanVar()
model_MobileNetV3Large = ttk.Checkbutton(mc_info, text="MobileNetV3Large", variable=variables["MobileNetV3Large"], onvalue = 1, offvalue = 0)
model_MobileNetV3Large.grid(row=3, column=5, sticky=(tk.W + tk.E))

# DenseNet 121
variables["DenseNet121"] = tk.BooleanVar()
model_DenseNet121 = ttk.Checkbutton(mc_info, text="DenseNet121", variable=variables["DenseNet121"], onvalue = 1, offvalue = 0)
model_DenseNet121.grid(row=3, column=6, sticky=(tk.W + tk.E))

# DenseNet 169
variables["DenseNet169"] = tk.BooleanVar()
model_DenseNet169 = ttk.Checkbutton(mc_info, text="DenseNet169", variable=variables["DenseNet169"], onvalue = 1, offvalue = 0)
model_DenseNet169.grid(row=3, column=7, sticky=(tk.W + tk.E))



# DenseNet 201
variables["DenseNet201"] = tk.BooleanVar()
model_DenseNet201 = ttk.Checkbutton(mc_info, text="DenseNet201", variable=variables["DenseNet201"], onvalue = 1, offvalue = 0)
model_DenseNet201.grid(row=4, column=0, sticky=(tk.W + tk.E))

# NASNet Mobile
variables["NASNetMobile"] = tk.BooleanVar()
model_NASNetMobile = ttk.Checkbutton(mc_info, text="NASNetMobile", variable=variables["NASNetMobile"], onvalue = 1, offvalue = 0)
model_NASNetMobile.grid(row=4, column=1, sticky=(tk.W + tk.E))

# NASNet Large
variables["NASNetLarge"] = tk.BooleanVar()
model_NASNetLarge = ttk.Checkbutton(mc_info, text="NASNetLarge", variable=variables["NASNetLarge"], onvalue = 1, offvalue = 0)
model_NASNetLarge.grid(row=4, column=2, sticky=(tk.W + tk.E))

# EfficientNet B0
variables["EfficientNetB0"] = tk.BooleanVar()
model_EfficientNetB0 = ttk.Checkbutton(mc_info, text="EfficientNetB0", variable=variables["EfficientNetB0"], onvalue = 1, offvalue = 0)
model_EfficientNetB0.grid(row=4, column=3, sticky=(tk.W + tk.E))

# EfficientNet B0 Version 2
variables["EfficientNetB0V2"] = tk.BooleanVar()
model_EfficientNetB0V2 = ttk.Checkbutton(mc_info, text="EfficientNetB0V2", variable=variables["EfficientNetB0V2"], onvalue = 1, offvalue = 0)
model_EfficientNetB0V2.grid(row=4, column=4, sticky=(tk.W + tk.E))

# EfficientNet B1
variables["EfficientNetB1"] = tk.BooleanVar()
model_EfficientNetB1 = ttk.Checkbutton(mc_info, text="EfficientNetB1", variable=variables["EfficientNetB1"], onvalue = 1, offvalue = 0)
model_EfficientNetB1.grid(row=4, column=5, sticky=(tk.W + tk.E))

# EfficientNet B1 Version 2
variables["EfficientNetB1V2"] = tk.BooleanVar()
model_EfficientNetB1V2 = ttk.Checkbutton(mc_info, text="EfficientNetB1V2", variable=variables["EfficientNetB1V2"], onvalue = 1, offvalue = 0)
model_EfficientNetB1V2.grid(row=4, column=6, sticky=(tk.W + tk.E))

# EfficientNet B2
variables["EfficientNetB2"] = tk.BooleanVar()
model_EfficientNetB2 = ttk.Checkbutton(mc_info, text="EfficientNetB2", variable=variables["EfficientNetB2"], onvalue = 1, offvalue = 0)
model_EfficientNetB2.grid(row=4, column=7, sticky=(tk.W + tk.E))



# EfficientNet B2 Version 2
variables["EfficientNetB2V2"] = tk.BooleanVar()
model_EfficientNetB2V2 = ttk.Checkbutton(mc_info, text="EfficientNetB2V2", variable=variables["EfficientNetB2V2"], onvalue = 1, offvalue = 0)
model_EfficientNetB2V2.grid(row=5, column=0, sticky=(tk.W + tk.E))

# EfficientNet B3
variables["EfficientNetB3"] = tk.BooleanVar()
model_EfficientNetB3 = ttk.Checkbutton(mc_info, text="EfficientNetB3", variable=variables["EfficientNetB3"], onvalue = 1, offvalue = 0)
model_EfficientNetB3.grid(row=5, column=1, sticky=(tk.W + tk.E))

# EfficientNet B3 Version 2
variables["EfficientNetB3V2"] = tk.BooleanVar()
model_EfficientNetB3V2 = ttk.Checkbutton(mc_info, text="EfficientNetB3V2", variable=variables["EfficientNetB3V2"], onvalue = 1, offvalue = 0)
model_EfficientNetB3V2.grid(row=5, column=2, sticky=(tk.W + tk.E))

# EfficientNet B4
variables["EfficientNetB4"] = tk.BooleanVar()
model_EfficientNetB4 = ttk.Checkbutton(mc_info, text="EfficientNetB4", variable=variables["EfficientNetB4"], onvalue = 1, offvalue = 0)
model_EfficientNetB4.grid(row=5, column=3, sticky=(tk.W + tk.E))

# EfficientNet B5
variables["EfficientNetB5"] = tk.BooleanVar()
model_EfficientNetB5 = ttk.Checkbutton(mc_info, text="EfficientNetB5", variable=variables["EfficientNetB5"], onvalue = 1, offvalue = 0)
model_EfficientNetB5.grid(row=5, column=4, sticky=(tk.W + tk.E))

# EfficientNet B6
variables["EfficientNetB6"] = tk.BooleanVar()
model_EfficientNetB6 = ttk.Checkbutton(mc_info, text="EfficientNetB6", variable=variables["EfficientNetB6"], onvalue = 1, offvalue = 0)
model_EfficientNetB6.grid(row=5, column=5, sticky=(tk.W + tk.E))

# EfficientNet B7
variables["EfficientNetB7"] = tk.BooleanVar()
model_EfficientNetB7 = ttk.Checkbutton(mc_info, text="EfficientNetB7", variable=variables["EfficientNetB7"], onvalue = 1, offvalue = 0)
model_EfficientNetB7.grid(row=5, column=6, sticky=(tk.W + tk.E))

# EfficientNet Version 2 Smalll
variables["EfficientNetV2Small"] = tk.BooleanVar()
model_EfficientNetV2Small = ttk.Checkbutton(mc_info, text="EfficientNetV2Small", variable=variables["EfficientNetV2Small"], onvalue = 1, offvalue = 0)
model_EfficientNetV2Small.grid(row=5, column=7, sticky=(tk.W + tk.E))



# EfficientNet Version 2 Medium
variables["EfficientNetV2Medium"] = tk.BooleanVar()
model_EfficientNetV2Medium = ttk.Checkbutton(mc_info, text="EfficientNetV2Medium", variable=variables["EfficientNetV2Medium"], onvalue = 1, offvalue = 0)
model_EfficientNetV2Medium.grid(row=6, column=0, sticky=(tk.W + tk.E))

# EfficientNet Version 2 Large
variables["EfficientNetV2Large"] = tk.BooleanVar()
model_EfficientNetV2Large = ttk.Checkbutton(mc_info, text="EfficientNetV2Large", variable=variables["EfficientNetV2Large"], onvalue = 1, offvalue = 0)
model_EfficientNetV2Large.grid(row=6, column=1, sticky=(tk.W + tk.E))

# ConvNetXt Tiny
variables["ConvNeXtTiny"] = tk.BooleanVar()
model_ConvNeXtTiny = ttk.Checkbutton(mc_info, text="ConvNeXtTiny", variable=variables["ConvNeXtTiny"], onvalue = 1, offvalue = 0)
model_ConvNeXtTiny.grid(row=6, column=2, sticky=(tk.W + tk.E))

# ConvNetXt Small
variables["ConvNeXtSmall"] = tk.BooleanVar()
model_ConvNeXtSmall = ttk.Checkbutton(mc_info, text="ConvNeXtSmall", variable=variables["ConvNeXtSmall"], onvalue = 1, offvalue = 0)
model_ConvNeXtSmall.grid(row=6, column=3, sticky=(tk.W + tk.E))

# ConvNetXt Base
variables["ConvNeXtBase"] = tk.BooleanVar()
model_ConvNeXtBase = ttk.Checkbutton(mc_info, text="ConvNeXtBase", variable=variables["ConvNeXtBase"], onvalue = 1, offvalue = 0)
model_ConvNeXtBase.grid(row=6, column=4, sticky=(tk.W + tk.E))

# ConvNetXt Large
variables["ConvNeXtLarge"] = tk.BooleanVar()
model_ConvNeXtLarge = ttk.Checkbutton(mc_info, text="ConvNeXtLarge", variable=variables["ConvNeXtLarge"], onvalue = 1, offvalue = 0)
model_ConvNeXtLarge.grid(row=6, column=5, sticky=(tk.W + tk.E))

# ConvNetXt XLarge
variables["ConvNeXtXLarge"] = tk.BooleanVar()
model_ConvNeXtXLarge = ttk.Checkbutton(mc_info, text="ConvNeXtXLarge", variable=variables["ConvNeXtXLarge"], onvalue = 1, offvalue = 0)
model_ConvNeXtXLarge.grid(row=6, column=6, sticky=(tk.W + tk.E))

# RegNetX002
variables["RegNetX002"] = tk.BooleanVar()
model_RegNetX002 = ttk.Checkbutton(mc_info, text="RegNetX002", variable=variables["RegNetX002"], onvalue = 1, offvalue = 0)
model_RegNetX002.grid(row=6, column=7, sticky=(tk.W + tk.E))



# RegNetY002
variables["RegNetY002"] = tk.BooleanVar()
model_RegNetY002 = ttk.Checkbutton(mc_info, text="RegNetY002", variable=variables["RegNetY002"], onvalue = 1, offvalue = 0)
model_RegNetY002.grid(row=7, column=0, sticky=(tk.W + tk.E))

# RegNetX004
variables["RegNetX004"] = tk.BooleanVar()
model_RegNetX004 = ttk.Checkbutton(mc_info, text="RegNetX004", variable=variables["RegNetX004"], onvalue = 1, offvalue = 0)
model_RegNetX004.grid(row=7, column=1, sticky=(tk.W + tk.E))

# RegNetY004
variables["RegNetY004"] = tk.BooleanVar()
model_RegNetY004 = ttk.Checkbutton(mc_info, text="RegNetY004", variable=variables["RegNetY004"], onvalue = 1, offvalue = 0)
model_RegNetY004.grid(row=7, column=2, sticky=(tk.W + tk.E))

# RegNetX006
variables["RegNetX006"] = tk.BooleanVar()
model_RegNetX006 = ttk.Checkbutton(mc_info, text="RegNetX006", variable=variables["RegNetX006"], onvalue = 1, offvalue = 0)
model_RegNetX006.grid(row=7, column=3, sticky=(tk.W + tk.E))

# RegNetY006
variables["RegNetY006"] = tk.BooleanVar()
model_RegNetY006 = ttk.Checkbutton(mc_info, text="RegNetY006", variable=variables["RegNetY006"], onvalue = 1, offvalue = 0)
model_RegNetY006.grid(row=7, column=4, sticky=(tk.W + tk.E))

# RegNetX008
variables["RegNetX008"] = tk.BooleanVar()
model_RegNetX008 = ttk.Checkbutton(mc_info, text="RegNetX008", variable=variables["RegNetX008"], onvalue = 1, offvalue = 0)
model_RegNetX008.grid(row=7, column=5, sticky=(tk.W + tk.E))

# RegNetY008
variables["RegNetY008"] = tk.BooleanVar()
model_RegNetY008 = ttk.Checkbutton(mc_info, text="RegNetY008", variable=variables["RegNetY008"], onvalue = 1, offvalue = 0)
model_RegNetY008.grid(row=7, column=6, sticky=(tk.W + tk.E))

# RegNetX016
variables["RegNetX016"] = tk.BooleanVar()
model_RegNetX016 = ttk.Checkbutton(mc_info, text="RegNetX016", variable=variables["RegNetX016"], onvalue = 1, offvalue = 0)
model_RegNetX016.grid(row=7, column=7, sticky=(tk.W + tk.E))



# RegNetY016
variables["RegNetY016"] = tk.BooleanVar()
model_RegNetY016 = ttk.Checkbutton(mc_info, text="RegNetY016", variable=variables["RegNetY016"], onvalue = 1, offvalue = 0)
model_RegNetY016.grid(row=8, column=0, sticky=(tk.W + tk.E))

# RegNetX032
variables["RegNetX032"] = tk.BooleanVar()
model_RegNetX032 = ttk.Checkbutton(mc_info, text="RegNetX032", variable=variables["RegNetX032"], onvalue = 1, offvalue = 0)
model_RegNetX032.grid(row=8, column=1, sticky=(tk.W + tk.E))

# RegNetY032
variables["RegNetY032"] = tk.BooleanVar()
model_RegNetY032 = ttk.Checkbutton(mc_info, text="RegNetY032", variable=variables["RegNetY032"], onvalue = 1, offvalue = 0)
model_RegNetY032.grid(row=8, column=2, sticky=(tk.W + tk.E))

# RegNetX040
variables["RegNetX040"] = tk.BooleanVar()
model_RegNetX040 = ttk.Checkbutton(mc_info, text="RegNetX040", variable=variables["RegNetX040"], onvalue = 1, offvalue = 0)
model_RegNetX040.grid(row=8, column=3, sticky=(tk.W + tk.E))

# RegNetY040
variables["RegNetY040"] = tk.BooleanVar()
model_RegNetY040 = ttk.Checkbutton(mc_info, text="RegNetY040", variable=variables["RegNetY040"], onvalue = 1, offvalue = 0)
model_RegNetY040.grid(row=8, column=4, sticky=(tk.W + tk.E))

# RegNetX064
variables["RegNetX064"] = tk.BooleanVar()
model_RegNetX064 = ttk.Checkbutton(mc_info, text="RegNetX064", variable=variables["RegNetX064"], onvalue = 1, offvalue = 0)
model_RegNetX064.grid(row=8, column=5, sticky=(tk.W + tk.E))

# RegNetY064
variables["RegNetY064"] = tk.BooleanVar()
model_RegNetY064 = ttk.Checkbutton(mc_info, text="RegNetY064", variable=variables["RegNetY064"], onvalue = 1, offvalue = 0)
model_RegNetY064.grid(row=8, column=6, sticky=(tk.W + tk.E))

# RegNetX080
variables["RegNetX080"] = tk.BooleanVar()
model_RegNetX080 = ttk.Checkbutton(mc_info, text="RegNetX080", variable=variables["RegNetX080"], onvalue = 1, offvalue = 0)
model_RegNetX080.grid(row=8, column=7, sticky=(tk.W + tk.E))



# RegNetY080
variables["RegNetY080"] = tk.BooleanVar()
model_RegNetY080 = ttk.Checkbutton(mc_info, text="RegNetY080", variable=variables["RegNetY080"], onvalue = 1, offvalue = 0)
model_RegNetY080.grid(row=9, column=0, sticky=(tk.W + tk.E))

# RegNetX120
variables["RegNetX120"] = tk.BooleanVar()
model_RegNetX120 = ttk.Checkbutton(mc_info, text="RegNetX120", variable=variables["RegNetX120"], onvalue = 1, offvalue = 0)
model_RegNetX120.grid(row=9, column=1, sticky=(tk.W + tk.E))

# RegNetY120
variables["RegNetY120"] = tk.BooleanVar()
model_RegNetY120 = ttk.Checkbutton(mc_info, text="RegNetY120", variable=variables["RegNetY120"], onvalue = 1, offvalue = 0)
model_RegNetY120.grid(row=9, column=2, sticky=(tk.W + tk.E))

# RegNetX160
variables["RegNetX160"] = tk.BooleanVar()
model_RegNetX160 = ttk.Checkbutton(mc_info, text="RegNetX160", variable=variables["RegNetX160"], onvalue = 1, offvalue = 0)
model_RegNetX160.grid(row=9, column=3, sticky=(tk.W + tk.E))

# RegNetY160
variables["RegNetY160"] = tk.BooleanVar()
model_RegNetY160 = ttk.Checkbutton(mc_info, text="RegNetY160", variable=variables["RegNetY160"], onvalue = 1, offvalue = 0)
model_RegNetY160.grid(row=9, column=4, sticky=(tk.W + tk.E))

# RegNetX320
variables["RegNetX320"] = tk.BooleanVar()
model_RegNetX320 = ttk.Checkbutton(mc_info, text="RegNetX320", variable=variables["RegNetX320"], onvalue = 1, offvalue = 0)
model_RegNetX320.grid(row=9, column=5, sticky=(tk.W + tk.E))

# RegNetY320
variables["RegNetY320"] = tk.BooleanVar()
model_RegNetY320 = ttk.Checkbutton(mc_info, text="RegNetY320", variable=variables["RegNetY320"], onvalue = 1, offvalue = 0)
model_RegNetY320.grid(row=9, column=6, sticky=(tk.W + tk.E))



def selectall():
    model_Xception.state(['selected'])
    variables["Xception"].set(1)

    model_VGG16.state(['selected'])
    variables["VGG16"].set(1)

    model_VGG19.state(['selected'])
    variables["VGG19"].set(1)

    model_ResNet50.state(['selected'])
    variables["ResNet50"].set(1)

    model_ResNet50V2.state(['selected'])
    variables["ResNet50V2"].set(1)
    
    model_ResNetRS50.state(['selected'])
    variables["ResNetRS50"].set(1)
    
    model_ResNet101.state(['selected'])
    variables["ResNet101"].set(1)

    model_ResNet101V2.state(['selected'])
    variables["ResNet101V2"].set(1)
    
    model_ResNetRS101.state(['selected'])
    variables["ResNetRS101"].set(1)
    
    model_ResNet152.state(['selected'])
    variables["ResNet152"].set(1)
    
    model_ResNet152V2.state(['selected'])
    variables["ResNet152V2"].set(1)
    
    model_ResNetRS152.state(['selected'])
    variables["ResNetRS152"].set(1)

    model_ResNetRS200.state(['selected'])
    variables["ResNetRS200"].set(1)
    
    model_ResNetRS270.state(['selected'])
    variables["ResNetRS270"].set(1)
    
    model_ResNetRS350.state(['selected'])
    variables["ResNetRS350"].set(1)
    
    model_ResNetRS420.state(['selected'])
    variables["ResNetRS420"].set(1)
    
    model_InceptionV3.state(['selected'])
    variables["InceptionV3"].set(1)
    
    model_InceptionResNetV2.state(['selected'])
    variables["InceptionResNetV2"].set(1)
    
    model_MobileNet.state(['selected'])
    variables["MobileNet"].set(1)
    
    model_MobileNetV2.state(['selected'])
    variables["MobileNetV2"].set(1)
    
    model_MobileNetV3Small.state(['selected'])
    variables["MobileNetV3Small"].set(1)

    model_MobileNetV3Large.state(['selected'])
    variables["MobileNetV3Large"].set(1)
    
    model_DenseNet121.state(['selected'])
    variables["DenseNet121"].set(1)

    model_DenseNet169.state(['selected'])
    variables["DenseNet169"].set(1)

    model_DenseNet201.state(['selected'])
    variables["DenseNet201"].set(1)

    model_NASNetMobile.state(['selected'])
    variables["NASNetMobile"].set(1)

    model_NASNetLarge.state(['selected'])
    variables["NASNetLarge"].set(1)
    
    model_EfficientNetB0.state(['selected'])
    variables["EfficientNetB0"].set(1)

    model_EfficientNetB0V2.state(['selected'])
    variables["EfficientNetB0V2"].set(1)

    model_EfficientNetB1.state(['selected'])
    variables["EfficientNetB1"].set(1)

    model_EfficientNetB1V2.state(['selected'])
    variables["EfficientNetB1V2"].set(1)

    model_EfficientNetB2.state(['selected'])
    variables["EfficientNetB2"].set(1)
    
    model_EfficientNetB2V2.state(['selected'])
    variables["EfficientNetB2V2"].set(1)

    model_EfficientNetB3.state(['selected'])
    variables["EfficientNetB3"].set(1)

    model_EfficientNetB3V2.state(['selected']) 
    variables["EfficientNetB3V2"].set(1)

    model_EfficientNetB4.state(['selected'])
    variables["EfficientNetB4"].set(1)

    model_EfficientNetB5.state(['selected'])
    variables["EfficientNetB5"].set(1)
    
    model_EfficientNetB6.state(['selected'])
    variables["EfficientNetB6"].set(1)

    model_EfficientNetB7.state(['selected'])
    variables["EfficientNetB7"].set(1)

    model_EfficientNetV2Small.state(['selected'])
    variables["EfficientNetV2Small"].set(1)

    model_EfficientNetV2Medium.state(['selected'])
    variables["EfficientNetV2Medium"].set(1)

    model_EfficientNetV2Large.state(['selected'])
    variables["EfficientNetV2Large"].set(1)
    
    model_ConvNeXtTiny.state(['selected'])
    variables["ConvNeXtTiny"].set(1)

    model_ConvNeXtSmall.state(['selected'])
    variables["ConvNeXtSmall"].set(1)

    model_ConvNeXtBase.state(['selected'])
    variables["ConvNeXtBase"].set(1)

    model_ConvNeXtLarge.state(['selected'])
    variables["ConvNeXtLarge"].set(1)

    model_ConvNeXtXLarge.state(['selected'])
    variables["ConvNeXtXLarge"].set(1)
    
    model_RegNetX002.state(['selected'])
    variables["RegNetX002"].set(1)
    
    model_RegNetY002.state(['selected'])
    variables["RegNetY002"].set(1)

    model_RegNetX004.state(['selected'])
    variables["RegNetX004"].set(1)

    model_RegNetY004.state(['selected'])
    variables["RegNetY004"].set(1)

    model_RegNetX006.state(['selected'])
    variables["RegNetX006"].set(1)

    model_RegNetY006.state(['selected'])
    variables["RegNetY006"].set(1)    

    model_RegNetX008.state(['selected'])
    variables["RegNetX008"].set(1)

    model_RegNetY008.state(['selected'])
    variables["RegNetY008"].set(1)

    model_RegNetX016.state(['selected'])
    variables["RegNetX016"].set(1)

    model_RegNetY016.state(['selected'])
    variables["RegNetY016"].set(1)
    
    model_RegNetX032.state(['selected'])
    variables["RegNetX032"].set(1)

    model_RegNetY032.state(['selected'])
    variables["RegNetY032"].set(1)
    
    model_RegNetX040.state(['selected'])
    variables["RegNetX040"].set(1)

    model_RegNetY040.state(['selected'])
    variables["RegNetY040"].set(1)  
    
    model_RegNetX064.state(['selected'])
    variables["RegNetX064"].set(1)

    model_RegNetY064.state(['selected'])
    variables["RegNetY064"].set(1)  
    
    model_RegNetX080.state(['selected'])
    variables["RegNetX080"].set(1)

    model_RegNetY080.state(['selected'])
    variables["RegNetY080"].set(1)  

    model_RegNetX120.state(['selected'])
    variables["RegNetX120"].set(1)

    model_RegNetY120.state(['selected'])
    variables["RegNetY120"].set(1)  

    model_RegNetX160.state(['selected'])
    variables["RegNetX160"].set(1)

    model_RegNetY160.state(['selected'])
    variables["RegNetY160"].set(1)  

    model_RegNetX320.state(['selected'])
    variables["RegNetX320"].set(1)

    model_RegNetY320.state(['selected'])
    variables["RegNetY320"].set(1)



def unselect():
    model_Xception.state(['!selected'])
    variables["Xception"].set(0)

    model_VGG16.state(['!selected'])
    variables["VGG16"].set(0)

    model_VGG19.state(['!selected'])
    variables["VGG19"].set(0)

    model_ResNet50.state(['!selected'])
    variables["ResNet50"].set(0)

    model_ResNet50V2.state(['!selected'])
    variables["ResNet50V2"].set(0)
    
    model_ResNetRS50.state(['!selected'])
    variables["ResNetRS50"].set(0)
    
    model_ResNet101.state(['!selected'])
    variables["ResNet101"].set(0)

    model_ResNet101V2.state(['!selected'])
    variables["ResNet101V2"].set(0)
    
    model_ResNetRS101.state(['!selected'])
    variables["ResNetRS101"].set(0)
    
    model_ResNet152.state(['!selected'])
    variables["ResNet152"].set(0)
    
    model_ResNet152V2.state(['!selected'])
    variables["ResNet152V2"].set(0)
    
    model_ResNetRS152.state(['!selected'])
    variables["ResNetRS152"].set(0)

    model_ResNetRS200.state(['!selected'])
    variables["ResNetRS200"].set(0)
    
    model_ResNetRS270.state(['!selected'])
    variables["ResNetRS270"].set(0)
    
    model_ResNetRS350.state(['!selected'])
    variables["ResNetRS350"].set(0)
    
    model_ResNetRS420.state(['!selected'])
    variables["ResNetRS420"].set(0)
    
    model_InceptionV3.state(['!selected'])
    variables["InceptionV3"].set(0)
    
    model_InceptionResNetV2.state(['!selected'])
    variables["InceptionResNetV2"].set(0)
    
    model_MobileNet.state(['!selected'])
    variables["MobileNet"].set(0)
    
    model_MobileNetV2.state(['!selected'])
    variables["MobileNetV2"].set(0)
    
    model_MobileNetV3Small.state(['!selected'])
    variables["MobileNetV3Small"].set(0)

    model_MobileNetV3Large.state(['!selected'])
    variables["MobileNetV3Large"].set(0)
    
    model_DenseNet121.state(['!selected'])
    variables["DenseNet121"].set(0)

    model_DenseNet169.state(['!selected'])
    variables["DenseNet169"].set(0)

    model_DenseNet201.state(['!selected'])
    variables["DenseNet201"].set(0)

    model_NASNetMobile.state(['!selected'])
    variables["NASNetMobile"].set(0)

    model_NASNetLarge.state(['!selected'])
    variables["NASNetLarge"].set(0)
    
    model_EfficientNetB0.state(['!selected'])
    variables["EfficientNetB0"].set(0)

    model_EfficientNetB0V2.state(['!selected'])
    variables["EfficientNetB0V2"].set(0)

    model_EfficientNetB1.state(['!selected'])
    variables["EfficientNetB1"].set(0)

    model_EfficientNetB1V2.state(['!selected'])
    variables["EfficientNetB1V2"].set(0)

    model_EfficientNetB2.state(['!selected'])
    variables["EfficientNetB2"].set(0)
    
    model_EfficientNetB2V2.state(['!selected'])
    variables["EfficientNetB2V2"].set(0)

    model_EfficientNetB3.state(['!selected'])
    variables["EfficientNetB3"].set(0)

    model_EfficientNetB3V2.state(['!selected']) 
    variables["EfficientNetB3V2"].set(0)

    model_EfficientNetB4.state(['!selected'])
    variables["EfficientNetB4"].set(0)

    model_EfficientNetB5.state(['!selected'])
    variables["EfficientNetB5"].set(0)
    
    model_EfficientNetB6.state(['!selected'])
    variables["EfficientNetB6"].set(0)

    model_EfficientNetB7.state(['!selected'])
    variables["EfficientNetB7"].set(0)

    model_EfficientNetV2Small.state(['!selected'])
    variables["EfficientNetV2Small"].set(0)

    model_EfficientNetV2Medium.state(['!selected'])
    variables["EfficientNetV2Medium"].set(0)

    model_EfficientNetV2Large.state(['!selected'])
    variables["EfficientNetV2Large"].set(0)
    
    model_ConvNeXtTiny.state(['!selected'])
    variables["ConvNeXtTiny"].set(0)

    model_ConvNeXtSmall.state(['!selected'])
    variables["ConvNeXtSmall"].set(0)

    model_ConvNeXtBase.state(['!selected'])
    variables["ConvNeXtBase"].set(0)

    model_ConvNeXtLarge.state(['!selected'])
    variables["ConvNeXtLarge"].set(0)

    model_ConvNeXtXLarge.state(['!selected'])
    variables["ConvNeXtXLarge"].set(0)
    
    model_RegNetX002.state(['!selected'])
    variables["RegNetX002"].set(0)
    
    model_RegNetY002.state(['!selected'])
    variables["RegNetY002"].set(0)

    model_RegNetX004.state(['!selected'])
    variables["RegNetX004"].set(0)

    model_RegNetY004.state(['!selected'])
    variables["RegNetY004"].set(0)

    model_RegNetX006.state(['!selected'])
    variables["RegNetX006"].set(0)

    model_RegNetY006.state(['!selected'])
    variables["RegNetY006"].set(0)    

    model_RegNetX008.state(['!selected'])
    variables["RegNetX008"].set(0)

    model_RegNetY008.state(['!selected'])
    variables["RegNetY008"].set(0)

    model_RegNetX016.state(['!selected'])
    variables["RegNetX016"].set(0)

    model_RegNetY016.state(['!selected'])
    variables["RegNetY016"].set(0)
    
    model_RegNetX032.state(['!selected'])
    variables["RegNetX032"].set(0)

    model_RegNetY032.state(['!selected'])
    variables["RegNetY032"].set(0)
    
    model_RegNetX040.state(['!selected'])
    variables["RegNetX040"].set(0)

    model_RegNetY040.state(['!selected'])
    variables["RegNetY040"].set(0)  
    
    model_RegNetX064.state(['!selected'])
    variables["RegNetX064"].set(0)

    model_RegNetY064.state(['!selected'])
    variables["RegNetY064"].set(0)  
    
    model_RegNetX080.state(['!selected'])
    variables["RegNetX080"].set(0)

    model_RegNetY080.state(['!selected'])
    variables["RegNetY080"].set(0)  

    model_RegNetX120.state(['!selected'])
    variables["RegNetX120"].set(0)

    model_RegNetY120.state(['!selected'])
    variables["RegNetY120"].set(0)  

    model_RegNetX160.state(['!selected'])
    variables["RegNetX160"].set(0)

    model_RegNetY160.state(['!selected'])
    variables["RegNetY160"].set(0)  

    model_RegNetX320.state(['!selected'])
    variables["RegNetX320"].set(0)

    model_RegNetY320.state(['!selected'])
    variables["RegNetY320"].set(0)

# Select all
variables["selectall"] = tk.BooleanVar()
ttk.Button(mc_info, text="Select All", command=selectall).grid(row=10, column=6, padx=5, pady=5, sticky=(tk.W + tk.E))
variables["unselect"] = tk.BooleanVar()
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
        
    elif(variables['strategie'].get()==2):
        optimizer2['state']='readonly'
        optimizer3['state']='disabled'
        
        loss2['state']='readonly'
        loss3['state']='disabled'
        
        epoch2['state']='readonly'
        epoch3['state']='disabled'
        
        lr2["state"] = 'readonly'
        lr3["state"] = 'disabled'
        
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


train_info = ttk.LabelFrame(mc, text='Training parameters')
train_info.grid(padx=5, pady=5, sticky=(tk.W + tk.E))
for i in range(5):
    train_info.columnconfigure(i, weight=1)
    
variables['strategie'] = tk.IntVar()    
ttk.Label(train_info, text="Training step(s)").grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W + tk.E))
ttk.Radiobutton(train_info, text='1 Step', value='1', variable=variables['strategie'], command=sel).grid(row=1, column=0, padx=5, pady=5, sticky=(tk.W + tk.E))
ttk.Radiobutton(train_info, text='2 Steps', value='2', variable=variables['strategie'], command=sel).grid(row=1, column=1, padx=5, pady=5, sticky=(tk.W + tk.E))
ttk.Radiobutton(train_info, text='3 Steps', value='3', variable=variables['strategie'], command=sel) .grid(row=1, column=2, padx=5, pady=5, sticky=(tk.W + tk.E))
variables['strategie'].set(1)

variables["multigpu"] = tk.IntVar()
ttk.Label(train_info, text="Parallelization").grid(row=0, column=3, padx=5, pady=5, sticky=(tk.W + tk.E))
multi_gpu = ttk.Checkbutton(train_info, text="Multi GPU", variable=variables["multigpu"], onvalue = 1, offvalue = 0)
multi_gpu.grid(row=1, column=3, padx=5, pady=5, sticky=(tk.W + tk.E))
multi_gpu.state(['disabled'])
variables["multigpu"].set(0)

if (numgpu > 1) :
    multi_gpu.state(['enabled'])
    multi_gpu.state(['selected'])
    variables["multigpu"].set(1)
    
variables['checkpoint'] = tk.IntVar()
ttk.Label(train_info, text="Save & Restore").grid(row=0, column=4, padx=5, pady=5, sticky=(tk.W + tk.E))
chkpoint = ttk.Checkbutton(train_info, text="Checkpoint", variable=variables["checkpoint"])
chkpoint.grid(row=1, column=4, padx=5, pady=5, sticky=(tk.W + tk.E))
chkpoint.state(['selected'])
variables['checkpoint'].set(1)

ttk.Separator(train_info, orient='horizontal').grid(row=2, columnspan=5, padx=5, pady=5, sticky=(tk.W + tk.E))

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

ttk.Label(train_info, text="Step 1").grid(row=4, column=0, padx=5, pady=5, sticky=(tk.W + tk.E))
ttk.Label(train_info, text="Step 2").grid(row=5, column=0, padx=5, pady=5, sticky=(tk.W + tk.E))
ttk.Label(train_info, text="Step 3").grid(row=6, column=0, padx=5, pady=5, sticky=(tk.W + tk.E))

variables['optimizer1'] = tk.StringVar()
variables['optimizer2'] = tk.StringVar()
variables['optimizer3'] = tk.StringVar()


ttk.Label(train_info, text="Optimizer").grid(row=3, column=1, padx=5, pady=5, sticky=(tk.W + tk.E))
optimizer1 = ttk.Combobox(train_info, values=listOptimizer, textvariable=variables['optimizer1'], state='readonly')
optimizer1.grid(row=4, column=1, padx=5, pady=5, sticky=(tk.W + tk.E))
optimizer1.current(2)
optimizer2 = ttk.Combobox(train_info, values=listOptimizer, textvariable=variables['optimizer2'], state='disabled')
optimizer2.grid(row=5, column=1, padx=5, pady=5, sticky=(tk.W + tk.E))
optimizer2.current(2)
optimizer3 = ttk.Combobox(train_info, values=listOptimizer, textvariable=variables['optimizer3'], state='disabled')
optimizer3.grid(row=6, column=1, padx=5, pady=5, sticky=(tk.W + tk.E))
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

variables['loos1'] = tk.StringVar()
variables['loos2'] = tk.StringVar()
variables['loos3'] = tk.StringVar()

ttk.Label(train_info, text="Loss").grid(row=3, column=2, padx=5, pady=5, sticky=(tk.W + tk.E))
loss1 = ttk.Combobox(train_info, values=listLoss, textvariable=variables['loos1'], state='reaonly')
loss1.grid(row=4, column=2, padx=5, pady=5, sticky=(tk.W + tk.E))
loss1.current(2)
loss2 = ttk.Combobox(train_info, values=listLoss, textvariable=variables['loos2'], state='disabled')
loss2.grid(row=5, column=2, padx=5, pady=5, sticky=(tk.W + tk.E))
loss2.current(2)
loss3 = ttk.Combobox(train_info, values=listLoss, textvariable=variables['loos3'], state='disabled')
loss3.grid(row=6, column=2, padx=5, pady=5, sticky=(tk.W + tk.E))
loss3.current(2)

listEpoch = list(range(1,501))

variables['epoch1'] = tk.StringVar()
variables['epoch2'] = tk.StringVar()
variables['epoch3'] = tk.StringVar()

ttk.Label(train_info, text="Epoh").grid(row=3, column=3, padx=5, pady=5, sticky=(tk.W + tk.E))
epoch1 = ttk.Combobox(train_info, values=listEpoch, textvariable=variables['epoch1'], state='readonly')
epoch1.grid(row=4, column=3, padx=5, pady=5, sticky=(tk.W + tk.E))
epoch1.current(9)
epoch2 = ttk.Combobox(train_info, values=listEpoch, textvariable=variables['epoch2'], state='disabled')
epoch2.grid(row=5, column=3, padx=5, pady=5, sticky=(tk.W + tk.E))
epoch2.current(9)
epoch3 = ttk.Combobox(train_info, values=listEpoch, textvariable=variables['epoch3'], state='disabled')
epoch3.grid(row=6, column=3, padx=5, pady=5, sticky=(tk.W + tk.E))
epoch3.current(49)

listlr = [0.1,0.01,0.001,0.0001,0.00001]

variables['lr1'] = tk.DoubleVar()
variables['lr2'] = tk.DoubleVar()
variables['lr3'] = tk.DoubleVar()

ttk.Label(train_info, text="Learning Rate").grid(row=3, column=4, padx=5, pady=5, sticky=(tk.W + tk.E))
lr1 = ttk.Combobox(train_info, values=listlr, textvariable=variables['lr1'], state='readonly')
lr1.grid(row=4, column=4, padx=5, pady=5, sticky=(tk.W + tk.E))
lr1.current(1)
lr2 = ttk.Combobox(train_info, values=listlr, textvariable=variables['lr2'], state='disabled')
lr2.grid(row=5, column=4, padx=5, pady=5, sticky=(tk.W + tk.E))
lr2.current(1)
lr3 = ttk.Combobox(train_info, values=listlr, textvariable=variables['lr3'], state='disabled')
lr3.grid(row=6, column=4, padx=5, pady=5, sticky=(tk.W + tk.E))
lr3.current(1)


# Output Section
output_info = ttk.LabelFrame(mc, text='Output')
output_info.grid(padx=5, pady=5, sticky=(tk.W + tk.E))
for i in range(5):
    output_info.columnconfigure(i, weight=1)
    
# Chechkbox Save mODEL
variables["savemodel"] = tk.IntVar()
savemodel = ttk.Checkbutton(output_info, text="Save Model", variable=variables['savemodel'], onvalue = 1, offvalue = 0)
savemodel.grid(row=0, column=0, sticky=(tk.W + tk.E))
savemodel.state(['selected'])
variables["savemodel"].set(1)

# Checkbox Training / Validation Graphs
variables["traingraph"] = tk.IntVar()
traingraph = ttk.Checkbutton(output_info, text="Training / Validation Graphs", variable=variables["traingraph"], onvalue = 1, offvalue = 0)
traingraph.grid(row=0, column=1, sticky=(tk.W + tk.E))
traingraph.state(['selected'])
variables["traingraph"].set(1)

# Confusion Matrix
variables["confmatrix"] = tk.IntVar()   
confmatrix = ttk.Checkbutton(output_info, text="Confusion Matrix", variable=variables["confmatrix"], onvalue = 1, offvalue = 0)
confmatrix.grid(row=0, column=2, sticky=(tk.W + tk.E))
confmatrix.state(['selected'])
variables["confmatrix"].set(1)

# Classification Report
variables["classreport"] = tk.IntVar()
classreport = ttk.Checkbutton(output_info, text="Classification Report", variable=variables["classreport"], onvalue = 1, offvalue = 0)
classreport.grid(row=0, column=3, sticky=(tk.W + tk.E))
classreport.state(['selected'])
variables["classreport"].set(1)

# Conversion of the model in TFLite
variables["tflite"] = tk.IntVar()
tflite = ttk.Checkbutton(output_info, text="TFLite Conversion", variable=variables["tflite"], onvalue = 1, offvalue = 0)
tflite.grid(row=0, column=4, sticky=(tk.W + tk.E))
tflite.state(['!selected'])



# Info Section
info_info = ttk.LabelFrame(mc, text='Info')
info_info.grid(padx=5, pady=5, sticky=(tk.W + tk.E))
for i in range(1):
    info_info.columnconfigure(i, weight=1)
    
ttk.Label(info_info, text="GPUs Available: " + str(numgpu) + " - Python: " + platform.python_version() + " - TensorFlow: " + tf.__version__ + " - Keras: "  + k.__version__ + " - Numpy: " + np.version.version + " - Pandas: " + pd.__version__ + " - Sklearn: " + sk.__version__ + " - Seaborn: " + sns.__version__ + "  - Matplotlib: " + mpl.__version__).grid(row=0, column=0,)



# Execution
exec_info = ttk.LabelFrame(mc, text='Execution')
exec_info.grid(padx=5, pady=5, sticky=(tk.W + tk.E))
for i in range(5):
    exec_info.columnconfigure(i, weight=1)

pgb = tk.IntVar()
pb = ttk.Progressbar(exec_info,  orient='horizontal', mode='determinate', variable=pgb).grid(row=0, column=1, columnspan=3, sticky=(tk.E + tk.W))


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
    
    lr1.current(1)
    lr2.current(1)
    lr3.current(1)
    
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
    
def training(strategie, multigpu, base_model, model_name, _optimizer1, loss1, _epoch1, _optimizer2, _loss2, _epoch2, _optimizer3, _loss3, _epoch3, ds_train, ds_valid, savemodel, traingraph, confmatrix, classreport, tflite):
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(output_dir+'/model/'+model_name+".tf", verbose=1, save_best_only=True),
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True
                                         ),
    ]
    
    callbacks2 = [
        tf.keras.callbacks.ModelCheckpoint(output_dir+'/model/'+model_name+".tf", verbose=1, save_best_only=True),
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
    ]
    
    print (model_name)
    
    # Multi GPU
    if (multigpu == 1): 

        strategy = tf.distribute.MirroredStrategy()
        
        with strategy.scope():
           
            base_model.trainable = False
            
            # Add a new classifier layers on top of the base model
            inputs = tf.keras.Input(shape=(img_height, img_width, 3))
            x = base_model(inputs, training=False)
            x = layers.GlobalAveragePooling2D()(x)
            outputs = layers.Dense(variables["classes"].get())(x) 
            model = tf.keras.Model(inputs, outputs)
            
            # Compile the model
            model.compile(optimizer=_optimizer1,
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])
        
        # Train the model
        hist = model.fit(ds_train, validation_data=ds_valid, epochs=epoch1, callbacks=callbacks)   
        
        if (strategie == 2):
            with strategy.scope():
                
                # Fine-tune the base model
                base_model.trainable = True
                
                model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), # Low learning rate for fine-tuning
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                              metrics=['accuracy'])
                
            hist2 = model.fit(ds_train, validation_data=ds_valid, epochs=epoch2, callbacks=callbacks2)    
                
        if (strategie == 3):
            with strategy.scope():
                # Compile the model
                model.compile(optimizer=optimizer2,
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                              metrics=['accuracy'])
         
            # Train the model
            hist2 = model.fit(ds_train, validation_data=ds_valid, epochs=epoch2, callbacks=callbacks)                
                
            
            with strategy.scope():
                
                # Fine-tune the base model
                base_model.trainable = True
                
                model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), # Low learning rate for fine-tuning
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                              metrics=['accuracy'])
                
            hist3 = model.fit(ds_train, validation_data=ds_valid, epochs=variables["epoch3"].get(), callbacks=callbacks2)            
        
    # CPU or single GPU   
    else:
        base_model.trainable = False
        # Add a new classifier layers on top of the base model
        inputs = tf.keras.Input(shape=(img_height, img_width, 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(variables["classes"].get())(x) 
        model = tf.keras.Model(inputs, outputs)
        
        # Compile the model
        model.compile(optimizer=_optimizer1,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
    
        # Train the model
        hist = model.fit(ds_train, validation_data=ds_valid, epochs=int(_epoch1), callbacks=callbacks) 

        if (strategie == 2):
                
            # Fine-tune the base model
            base_model.trainable = True
            
            model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), # Low learning rate for fine-tuning
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])
            
            hist2 = model.fit(ds_train, validation_data=ds_valid, epochs=int(_epoch2), callbacks=callbacks2)    
                
        if (strategie == 3):

            # Compile the model
            model.compile(optimizer=optimizer3,
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])
         
            # Train the model
            hist2 = model.fit(ds_train, validation_data=ds_valid, epochs=int(_epoch2), callbacks=callbacks)                
                
            # Fine-tune the base model
            base_model.trainable = True
            
            model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), # Low learning rate for fine-tuning
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])
                
            hist3 = model.fit(ds_train, validation_data=ds_valid, epochs=int(_epoch3), callbacks=callbacks2)            

    
    #Output
    if (savemodel == 1):
        model_json = model.to_json()
        with open(output_dir+'/model/'+model_name+'.json', 'w') as json_file:
            json_file.write(model_json)        

    if (traingraph == 1):
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
    
    if (variables["Xception"].get() == 1):
        total = total + 1
    
    if (variables["VGG16"].get() == 1):
        total = total + 1
        
    if (variables["VGG19"].get() == 1):
        total = total + 1
       
    if (variables["ResNet50"].get() == 1):
        total = total + 1 
        
    if (variables["ResNet50V2"].get() == 1):
        total = total + 1
        
    if (variables["ResNetRS50"].get() == 1):
        total = total + 1        
     
    if (variables["ResNet101"].get() == 1):
        total = total + 1
        
    if (variables["ResNet101V2"].get() == 1):
        total = total + 1
        
    if (variables["ResNetRS101"].get() == 1):
        total = total + 1 
    
    if (variables["ResNet152"].get() == 1):
        total = total + 1
        
    if (variables["ResNet152V2"].get() == 1):
        total = total + 1
        
    if (variables["ResNetRS152"].get() == 1):
        total = total + 1 
       
    if (variables["ResNetRS200"].get() == 1):
        total = total + 1 
        
    if (variables["ResNetRS270"].get() == 1):
        total = total + 1 
        
    if (variables["ResNetRS350"].get() == 1):
        total = total + 1 
        
    if (variables["ResNetRS420"].get() == 1):
        total = total + 1 
        
    if (variables["InceptionV3"].get() == 1):
        total = total + 1
        
    if (variables["InceptionResNetV2"].get() == 1):
        total = total + 1
    
    if (variables["MobileNet"].get() == 1):
        total = total + 1       
    
    if (variables["MobileNetV2"].get() == 1):
        total = total + 1        
        
    if (variables["MobileNetV3Small"].get() == 1):
        total = total + 1        
       
    if (variables["MobileNetV3Large"].get() == 1):
        total = total + 1       
        
    if (variables["DenseNet121"].get() == 1):
        total = total + 1
        
    if (variables["DenseNet169"].get() == 1):
        total = total + 1       
        
    if (variables["DenseNet201"].get() == 1):
        total = total + 1      
        
    if (variables["NASNetMobile"].get() == 1):
        total = total + 1        
    
    if (variables["NASNetLarge"].get() == 1):
        total = total + 1 
        
    if (variables["EfficientNetB0"].get() == 1):
        total = total + 1        
    
    if (variables["EfficientNetB0V2"].get() == 1):
        total = total + 1        
        
    if (variables["EfficientNetB1"].get() == 1):
        total = total + 1        
        
    if (variables["EfficientNetB1V2"].get() == 1):
        total = total + 1        
        
    if (variables["EfficientNetB2"].get() == 1):
        total = total + 1        
        
    if (variables["EfficientNetB2V2"].get() == 1):
        total = total + 1   
        
    if (variables["EfficientNetB3"].get() == 1):
        total = total + 1        
        
    if (variables["EfficientNetB3V2"].get() == 1):
        total = total + 1  
        
    if (variables["EfficientNetB4"].get() == 1):
        total = total + 1 
        
    if (variables["EfficientNetB5"].get() == 1):
        total = total + 1 

    if (variables["EfficientNetB6"].get() == 1):
        total = total + 1 

    if (variables["EfficientNetB7"].get() == 1):
        total = total + 1 
        
    if (variables["EfficientNetV2Small"].get() == 1):  
        total = total + 1 
        
    if (variables["EfficientNetV2Medium"].get() == 1):
        total = total + 1 
     
    if (variables["EfficientNetV2Large"].get() == 1):
        total = total + 1 
    
    if (variables["ConvNeXtTiny"].get() == 1):
        total = total + 1        
        
    if (variables["ConvNeXtSmall"].get() == 1):
        total = total + 1        
        
    if (variables["ConvNeXtBase"].get() == 1):
        total = total + 1    
        
    if (variables["ConvNeXtLarge"].get() == 1):
        total = total + 1        
        
    if (variables["ConvNeXtXLarge"].get() == 1):
        total = total + 1  
        
    if (variables["RegNetX002"].get() == 1):
        total = total + 1
        
    if (variables["RegNetY002"].get() == 1):
        total = total + 1
        
    if (variables["RegNetX004"].get() == 1):
        total = total + 1
        
    if (variables["RegNetY004"].get() == 1):
        total = total + 1
        
    if (variables["RegNetX006"].get() == 1):
        total = total + 1
        
    if (variables["RegNetY006"].get() == 1):
        total = total + 1
    
    if (variables["RegNetX008"].get() == 1):
        total = total + 1
        
    if (variables["RegNetY008"].get() == 1):
        total = total + 1

    if (variables["RegNetX016"].get() == 1):
        total = total + 1
        
    if (variables["RegNetY016"].get() == 1):
        total = total + 1

    if (variables["RegNetX032"].get() == 1):
        total = total + 1
        
    if (variables["RegNetY032"].get() == 1):
        total = total + 1
 
    if (variables["RegNetX040"].get() == 1):
        total = total + 1
        
    if (variables["RegNetY040"].get() == 1):
        total = total + 1  
        
    if (variables["RegNetX064"].get() == 1):
        total = total + 1
        
    if (variables["RegNetY064"].get() == 1):
        total = total + 1
        
    if (variables["RegNetX080"].get() == 1):
        total = total + 1
        
    if (variables["RegNetY080"].get() == 1):
        total = total + 1
 
    if (variables["RegNetX120"].get() == 1):
        total = total + 1
        
    if (variables["RegNetY120"].get() == 1):
        total = total + 1   
        
    if (variables["RegNetX160"].get() == 1):
        total = total + 1
        
    if (variables["RegNetY160"].get() == 1):
        total = total + 1
        
    if (variables["RegNetX320"].get() == 1):
        total = total + 1
        
    if (variables["RegNetY320"].get() == 1):
        total = total + 1
 
    ### Xception Model ##    
    if (variables["Xception"].get() == 1):
        model_name = "Xception"  
        
        base_model = Xception(input_shape=(img_height, img_width, int(variables["channel"].get())),
                              include_top=False,
                              weights='imagenet')
            
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()
        
    
    ### VGG16 Model ###
    if (variables["VGG16"].get() == 1):
        model_name = "VGG16"
        
        base_model = VGG16(input_shape=(img_height, img_width, int(variables["channel"].get())),
                           include_top=False,
                           weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()   
    
    
    ### VGG19 model ###
    if (variables["VGG19"].get() == 1):
        model_name = "VGG19"
        
        base_model = VGG19(input_shape=(img_height, img_width, int(variables["channel"].get())),
                           include_top=False,
                           weights='imagenet')  
        
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())

        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()
    
    
    ### ResNet50 ###
    if (variables["ResNet50"].get() == 1):
        model_name = "ResNet50"
        
        base_model = ResNet50(input_shape=(img_height, img_width, int(variables["channel"].get())),
                              include_top=False,
                              weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())

        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()
    

    ### ResNet50 V2 ###
    if (variables["ResNet50V2"].get() == 1):
        model_name = "ResNet50_V2"
        
        base_model = ResNet50V2(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                include_top=False,
                                weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())

        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()
        
        
    ### ResNetRS50 ###
    if (variables["ResNetRS50"].get() == 1):
        model_name = "ResNetRS50"
        
        base_model = ResNetRS50(input_shape=(img_height, img_width, int(variables["channel"].get())),
                              include_top=False,
                              weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())

        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()
    
            
    ### ResNet101 ###
    if (variables["ResNet101"].get() == 1):
        model_name = "ResNet101"
        
        base_model = ResNet101(input_shape=(img_height, img_width, int(variables["channel"].get())),
                               include_top=False,
                               weights='imagenet')
        
        
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
     
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()   
        
  
    ### ResNet101 V2 ###
    if (variables["ResNet101V2"].get() == 1):
        model_name = "ResNet101_V2"
        
        base_model = ResNet101V2(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                 include_top=False,
                                 weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())

        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()
        
    
    ### ResNetRS101 ###
    if (variables["ResNetRS101"].get() == 1):
        model_name = "ResNetRS101"
        
        base_model = ResNetRS101(input_shape=(img_height, img_width, int(variables["channel"].get())),
                              include_top=False,
                              weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())

        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()
        
        
    ### ResNet152 ###
    if (variables["ResNet152"].get() == 1):
        model_name = "ResNet152"
        base_model = ResNet152(input_shape=(img_height, img_width, int(variables["channel"].get())),
                               include_top=False,
                               weights='imagenet')
        
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()
        
        
    ### ResNet152 V2 ###
    if (variables["ResNet152V2"].get() == 1):
        model_name = "ResNet152_V2"
        base_model = ResNet152V2(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                 include_top=False,
                                 weights='imagenet')
        
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()   

 
    ### ResNetRS 152 ###
    if (variables["ResNetRS152"].get() == 1):
        model_name = "ResNetRS152"
        base_model = ResNetRS152(input_shape=(img_height, img_width, int(variables["channel"].get())),
                              include_top=False,
                              weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())

        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()


    ### ResNetRS 200 ###
    if (variables["ResNetRS200"].get() == 1):
        model_name = "ResNetRS200"
        base_model = ResNetRS200(input_shape=(img_height, img_width, int(variables["channel"].get())),
                              include_top=False,
                              weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())

        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()
  
    
    ### ResNetRS 270 ###
    if (variables["ResNetRS270"].get() == 1):
        model_name = "ResNetRS270"
        base_model = ResNetRS270(input_shape=(img_height, img_width, int(variables["channel"].get())),
                              include_top=False,
                              weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())

        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()
    
    
    ### ResNetRS 350 ###
    if (variables["ResNetRS350"].get() == 1):
        model_name = "ResNetRS350"
        base_model = ResNetRS350(input_shape=(img_height, img_width, int(variables["channel"].get())),
                              include_top=False,
                              weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())

        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()    


    ### ResNetRS 420 ###
    if (variables["ResNetRS420"].get() == 1):
        model_name = "ResNetRS420"
        base_model = ResNetRS420(input_shape=(img_height, img_width, int(variables["channel"].get())),
                              include_top=False,
                              weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())

        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()
    
    
    ### Inception V3 ###
    if (variables["InceptionV3"].get() == 1):
        model_name = "Inception_V3"
        base_model = InceptionV3(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                 include_top=False,
                                 weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()   

    
    ### InceptionResNet V2 ###
    if (variables["InceptionResNetV2"].get() == 1):
        model_name = "InceptionResNet_V2"
        base_model = InceptionResNetV2(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                       include_top=False,
                                       weights='imagenet')
        
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()
    
    
    ### MobileNet ###
    if (variables["MobileNet"].get() == 1):
        model_name = "MobileNet"
        base_model = MobileNet(input_shape=(img_height, img_width, int(variables["channel"].get())),
                               include_top=False,
                               weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()
    
    
    ### MobileNet V2 ###
    if (variables["MobileNetV2"].get() == 1):
        model_name = "MobileNet_V2"
        base_model = MobileNetV2(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                 include_top=False,
                                 weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()
    
    
    ### MobileNet V3 Small ###
    if (variables["MobileNetV3Small"].get() == 1):
        model_name = "MobileNet_V3_Small"
        base_model = MobileNetV3Small(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                      include_top=False,
                                      weights='imagenet')
        
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()
    
    
    ### MobileNet V3 Large ###
    if (variables["MobileNetV3Large"].get() == 1):
        model_name = "MobileNet_V3_Large"

        base_model = MobileNetV3Large(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                      include_top=False,
                                      weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())

        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()
    
    
    ### DenseNet 121 ###
    if (variables["DenseNet121"].get() == 1):
        model_name = "DenseNet121"
        base_model = DenseNet121(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                 include_top=False,
                                 weights='imagenet')
        
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()
    
    
    ### DenseNet 169 ###
    if (variables["DenseNet169"].get() == 1):
        model_name = "DenseNet169"
        base_model = DenseNet169(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                 include_top=False,
                                 weights='imagenet')
        
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()   
        
        
    ### DenseNet 201 ###
    if (variables["DenseNet201"].get() == 1):
        model_name = "DenseNet201"
        base_model = DenseNet201(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                 include_top=False,
                                 weights='imagenet')
        
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()
    
    
    ### NASNetMobile ###
    if (variables["NASNetMobile"].get() == 1):
        model_name = "NASNetMobile"
        base_model = NASNetMobile(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                  include_top=False,
                                  weights='imagenet')
        
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()
    
    
    ### NASNetLarge ###
    if (variables["NASNetLarge"].get() == 1):
        model_name = "NASNetLarge"
        base_model = NASNetLarge(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()
    
    
    ### EfficientNetB0 ###
    if (variables["EfficientNetB0"].get() == 1):
        model_name = "EfficientNet_B0"
        base_model = EfficientNetB0(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()
    
    
    ### EfficientNetB0 V2 ###
    if (variables["EfficientNetB0V2"].get() == 1):
        model_name = "EfficientNet_B0_V2"
        base_model = EfficientNetV2B0(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()


    ### EfficientNetB1 ###
    if (variables["EfficientNetB1"].get() == 1):
        model_name = "EfficientNet_B1"
        base_model = EfficientNetB1(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()
    
  
    ### EfficientNetB1 V2 ###
    if (variables["EfficientNetB1V2"].get() == 1):
        model_name = "EfficientNet_B1_V2"
        base_model = EfficientNetV2B1(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()
    
    
    ### EfficientNetB2 ###
    if (variables["EfficientNetB2"].get() == 1):
        model_name = "EfficientNet_B2"
        base_model = EfficientNetB2(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()
    

    ### EfficientNetB2 V2 ###
    if (variables["EfficientNetB2V2"].get() == 1):
        model_name = "EfficientNet_B2_V2"
        base_model = EfficientNetV2B2(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()
    
        
    ### EfficientNetB3 ###
    if (variables["EfficientNetB3"].get() == 1):
        model_name = "EfficientNet_B3"
        base_model = EfficientNetB3(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()
    
  
    ### EfficientNetB3 V2 ###
    if (variables["EfficientNetB3V2"].get() == 1):
        model_name = "EfficientNet_B3_V2"
        base_model = EfficientNetV2B3(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()
    

    ### EfficientNetB4 ###
    if (variables["EfficientNetB4"].get() == 1):
        model_name = "EfficientNet_B4"
        base_model = EfficientNetB4(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()
    
    
    ### EfficientNetB5 ###
    if (variables["EfficientNetB5"].get() == 1):
        model_name = "EfficientNet_B5"
        base_model = EfficientNetB5(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
      
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()
    
    
    ### EfficientNetB6 ###
    if (variables["EfficientNetB6"].get() == 1):
        model_name = "EfficientNet_B6"
        base_model = EfficientNetB6(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()
    
    
    ### EfficientNetB7 ###
    if (variables["EfficientNetB7"].get() == 1):
        model_name = "EfficientNet_B7"
        base_model = EfficientNetB7(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
        
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()
    
        
    ### EfficientNet2S ###
    if (variables["EfficientNetV2Small"].get() == 1): 
        model_name = "EfficientNet_V2_Small"
        base_model = EfficientNetV2S(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()
    
    
    ### EfficientNet2M ###
    if (variables["EfficientNetV2Medium"].get() == 1):
        model_name = "EfficientNet_V2_Medium"
        base_model = EfficientNetV2M(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()
    
    
    ### EfficientNet2L ###
    if (variables["EfficientNetV2Large"].get() == 1):
        model_name = "EfficientNet_V2_Large"
        base_model = EfficientNetV2L(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()
    
    
    ### ConvNeXtTiny ###
    if (variables["ConvNeXtTiny"].get() == 1):
        model_name = "ConvNeXtTiny"
        base_model = ConvNeXtTiny(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()
    
    
    ### ConvNeXtSmall ###
    if (variables["ConvNeXtSmall"].get() == 1):
        model_name = "ConvNeXtSmall"
        base_model = ConvNeXtSmall(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()
    
    
    ### ConvNeXtBase ###
    if (variables["ConvNeXtBase"].get() == 1):
        model_name = "ConvNeXtBase"
        base_model = ConvNeXtBase(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
        
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()
    
    
    ### ConvNeXtLarge ###
    if (variables["ConvNeXtLarge"].get() == 1):
        model_name = "ConvNeXtLarge"
        base_model = ConvNeXtLarge(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()
    
    
    ### ConvNeXtXLarge ###
    if (variables["ConvNeXtXLarge"].get() == 1):
        model_name = "ConvNeXtXLarge"
        base_model = ConvNeXtXLarge(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()
        
    ### RegNetX002 ###
    if (variables["RegNetX002"].get() == 1):
        model_name = "RegNetX002"
        base_model = RegNetX002(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()   
    
    ### RegNetY002 ###
    if (variables["RegNetY002"].get() == 1):
        model_name = "RegNetY002"
        base_model = RegNetY002(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update() 
        
    ### RegNetX004 ###
    if (variables["RegNetX004"].get() == 1):
        model_name = "RegNetX004"
        base_model = RegNetX004(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()   
    
    ### RegNetY004 ###
    if (variables["RegNetY004"].get() == 1):
        model_name = "RegNetY004"
        base_model = RegNetY004(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()
        
    ### RegNetX006 ###
    if (variables["RegNetX006"].get() == 1):
        model_name = "RegNetX006"
        base_model = RegNetX006(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()   
    
    ### RegNetY006 ###
    if (variables["RegNetY006"].get() == 1):
        model_name = "RegNetY006"
        base_model = RegNetY006(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()         

    ### RegNetX008 ###
    if (variables["RegNetX008"].get() == 1):
        model_name = "RegNetX008"
        base_model = RegNetX008(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()   
    
    ### RegNetY008 ###
    if (variables["RegNetY008"].get() == 1):
        model_name = "RegNetY008"
        base_model = RegNetY008(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update() 
        
    ### RegNetX016 ###
    if (variables["RegNetX016"].get() == 1):
        model_name = "RegNetX016"
        base_model = RegNetX016(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()   
    
    ### RegNetY016 ###
    if (variables["RegNetY016"].get() == 1):
        model_name = "RegNetY016"
        base_model = RegNetY016(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()

    ### RegNetX032 ###
    if (variables["RegNetX032"].get() == 1):
        model_name = "RegNetX032"
        base_model = RegNetX032(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()   
    
    ### RegNetY032 ###
    if (variables["RegNetY032"].get() == 1):
        model_name = "RegNetY032"
        base_model = RegNetY032(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()

    ### RegNetX040 ###
    if (variables["RegNetX040"].get() == 1):
        model_name = "RegNetX040"
        base_model = RegNetX040(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()   
    
    ### RegNetY040 ###
    if (variables["RegNetY040"].get() == 1):
        model_name = "RegNetY040"
        base_model = RegNetY040(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()

    ### RegNetX064 ###
    if (variables["RegNetX064"].get() == 1):
        model_name = "RegNetX064"
        base_model = RegNetX064(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()   
    
    ### RegNetY064 ###
    if (variables["RegNetY064"].get() == 1):
        model_name = "RegNetY064"
        base_model = RegNetY064(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()
        
    ### RegNetX080 ###
    if (variables["RegNetX080"].get() == 1):
        model_name = "RegNetX080"
        base_model = RegNetX080(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()   
    
    ### RegNetY080 ###
    if (variables["RegNetY080"].get() == 1):
        model_name = "RegNetY080"
        base_model = RegNetY080(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()  
        
    ### RegNetX120 ###
    if (variables["RegNetX120"].get() == 1):
        model_name = "RegNetX120"
        base_model = RegNetX120(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()   
    
    ### RegNetY120 ###
    if (variables["RegNetY120"].get() == 1):
        model_name = "RegNetY120"
        base_model = RegNetY120(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()
        
    ### RegNetX160 ###
    if (variables["RegNetX160"].get() == 1):
        model_name = "RegNetX160"
        base_model = RegNetX160(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()   
    
    ### RegNetY160 ###
    if (variables["RegNetY160"].get() == 1):
        model_name = "RegNetY160"
        base_model = RegNetY160(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update() 
 
    ### RegNetX320 ###
    if (variables["RegNetX320"].get() == 1):
        model_name = "RegNetX320"
        base_model = RegNetX320(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()   
    
    ### RegNetY080 ###
    if (variables["RegNetY320"].get() == 1):
        model_name = "RegNetY320"
        base_model = RegNetY320(input_shape=(img_height, img_width, int(variables["channel"].get())),
                                    include_top=False,
                                    weights='imagenet')
    
        training(variables['strategie'].get(), variables["multigpu"].get(), base_model, model_name, variables['optimizer1'].get(), variables['loos1'].get(), variables['epoch1'].get(), variables['optimizer2'].get(), variables['loos2'].get(), variables['epoch2'].get(), variables['optimizer3'].get(), variables['loos3'].get(), variables['epoch3'].get(), train_ds, val_ds, variables["savemodel"].get(), variables["traingraph"].get(), variables["confmatrix"].get(), variables["classreport"].get(), variables["tflite"].get())
    
        cpt = cpt + 1
        pgb.set(round((cpt/total)*100))
        mc.update()    
 
    print ("End")

# Execution 
ttk.Button(exec_info, text="Reset", command=reset).grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W + tk.E))
ttk.Button(exec_info, text="Run", command=run).grid(row=0, column=4, padx=5, pady=5, sticky=(tk.W + tk.E))


# Show the window 
root.mainloop()