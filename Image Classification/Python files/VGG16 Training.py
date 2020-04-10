
# coding: utf-8

# # VGG 16  Training

# In[ ]:


from PIL import Image
import PIL
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ### Function which helps to find the average shape of images in training folder
# 

# In[ ]:


def CheckDimsImage(folder_loc):
    path = folder_loc
    os.chdir(path)
    lis = os.listdir()
    width1 = []
    height1 = []
    print("Checking size of all images --> \n\n\n")
    for l in lis:
        os.chdir(path+"//"+l)
        images = os.listdir()
        for img in images:
            image = PIL.Image.open(img)
            width, height = image.size
            width1.append(width)
            height1.append(height)
        os.chdir('..')    
        dictq  = {'width' : width1,
                 'height' : height1}

    df = pd.DataFrame(dictq, index = None)
    df['mix'] = list(zip(df.width, df.height))
    print("Number of unique value of diesion of images -->\n\n\n")
    print(df.nunique())
    print(df.head())
    avg_height = int(np.average(height1))
    avg_width = int(np.average(width1))
    print("Average height -->",avg_height)
    print("Average width -->",avg_width)
        


# ### Path of training folder

# In[ ]:


path = '//kaggle//input//alien-vs-predator-images//data//train'


# In[ ]:


CheckDimsImage(path)


# ### Function which helps to find the nuber of images in each class and detrmine the training weight according to it

# In[ ]:


def NumfileEachClass(folder_loc):
    path = folder_loc
    os.chdir(path)
    lis = os.listdir()
    print("Show the distribution of images in all classes-> \n\n\n")
    classf = []
    number = []
    for l in lis:
        os.chdir(path+"//"+l)
        images = os.listdir()
        num_images = int(len(images))
        classf.append(l)
        number.append(num_images)
        os.chdir('..') 
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    class_f = classf
    num = number
    ax.bar(classf,number)
    plt.show()
    labels_count = dict()
    for img_class in [ic for ic in os.listdir(path) if ic[0] != '.']:
        labels_count[img_class] = len(os.listdir(path +'//'+ img_class))
    total_count = sum(labels_count.values())
    class_weights = {cls: total_count / count for cls, count in 
                     enumerate(labels_count.values())}

    return class_weights


# In[ ]:


n = NumfileEachClass(path)


# ### Importing libraries and defining parameters

# In[ ]:


import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout,Conv2D,MaxPooling2D
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import load_model

height = 96
width = 96
input_sh = (height, width, 3)
rotation = 40
width_shift = 0.2
height_shift = 0.2
scale = 1/255
shear = 0.2
zoom = 0.2
horizontal = True
fill = 'nearest'
validation = 0.2
batch = 16
dropout_num = 0.1
target = (height, width)
hidden = 256
classes = 2
loss_param ='categorical_crossentropy'
epoch = 20
patience_param = 5         
output_path = '//kaggle//working'


# ### Main training function

# In[ ]:


def VGGModel(trainable=None):
    
    
    model = keras.applications.VGG16(
    include_top=False, 
    weights='imagenet',
    input_shape=input_shape
     )
    
    new_model = Sequential()
    
    for l in model.layers:
        new_model.add(l)
    print(new_model.summary()) 
    # Lock the CONV lyers from gtraining
    if trainable is not None:
        for layer in new_model.layers[:trainable]:
            layer.trainable = False

    top = Sequential()
    top.add(Flatten(input_shape=new_model.output_shape[1:]))
    top.add(Dense(hidden_top, activation='relu', name='Dense_Intermediate_1'))
    top.add(Dropout(dropout))
    top.add(Dense(2*hidden_top, activation='relu', name='Dense_Intermediate_2'))
    top.add(Dropout(dropout, name='Dropout_Regularization'))
    if num_class == 1:
        top.add(Dense(num_class, activation='sigmoid', name='Output'))
    else:
        top.add(Dense(num_class, activation='softmax', name='Output'))
    # Concanate VGG16 layers and FC layers 
    new_model.add(top)    
    print(new_model.summary())
    
    # Preproceesing the dataset into keras feedable format
    train_datagen = ImageDataGenerator(
        rotation_range = rotation,
        width_shift_range = width_shift,
        height_shift_range= height_shift,
        rescale= scale,
        shear_range= shear,
        zoom_range= zoom,
        horizontal_flip= horizontal,
        fill_mode=fill,
        validation_split=validation
    )
    test_datagen = ImageDataGenerator(
        rescale= scale,
    )

    train_generator = train_datagen.flow_from_directory(
        path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
    )
    validation_generator = train_datagen.flow_from_directory(
        path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    # Compile the model

    new_model.compile(
        optimizer=RMSprop(),                       # Change the parameters according to need
        loss=loss,
        metrics=['accuracy']
    )


    new_model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator.filenames) // batch_size,
        epochs=epoch,
        validation_data=validation_generator,
        validation_steps=len(train_generator.filenames) // batch_size,
        class_weight=n,
        callbacks=[
            EarlyStopping(patience=patience, restore_best_weights=True),
            ReduceLROnPlateau(patience=patience)
        ])
    os.chdir(output_path)    
    model.save("model.h5")    
    


# In[ ]:


VGGModel(trainable=16)   # this means top 15 layers weight are non trainable


# In[ ]:


VGGModel()      # this means all parameters are trainable

