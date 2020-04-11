
# coding: utf-8

# # Xception  Training

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


# ### Function which helps to find the nuber of images in each class and determine the training weight according to it

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
from keras.layers import Flatten, Dense, Dropout,Conv2D,MaxPooling2D
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import load_model
from keras import models, regularizers, layers, optimizers, losses, metrics
from keras.models import Sequential
from keras.utils import np_utils, to_categorical
from keras.preprocessing import image
from keras.applications import Xception
from keras.models import Model

height = 224
width = 224
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
batch = 32
dropout_num = 0.5
target = (height, width)
hidden = 512
classes = 2
loss_param ='categorical_crossentropy'
epoch = 20
patience_param = 5        
lroptimizer = 2e-5
output_path = '//kaggle//working'


# ### Main training function

# In[ ]:


def Xception_Net(trainable=None):
    
    # Preprocessing the dataset into keras feedable format
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
        target_size=target,
        batch_size=batch,
        class_mode='categorical',
        subset='training',
    )
    validation_generator = train_datagen.flow_from_directory(
        path,
        target_size=target,
        batch_size=batch,
        class_mode='categorical',
        subset='validation'
    )
    
    
    # Loading the ResNet50 Model
    
    xception = Xception(include_top=False, weights='imagenet', input_shape=input_sh)
    output = xception.layers[-1].output
    output = keras.layers.Flatten()(output)
    xception = Model(xception.input, output=output)
    print(xception.summary())
    print('\n\n\n')
    # If you chose not for fine tuning
    if trainable is None:
        model = Sequential()
        model.add(xception)
        model.add(Dense(hidden, activation='relu', input_dim=input_sh))
        model.add(Dropout(dropout_num))
        model.add(Dense(hidden, activation='relu'))
        model.add(Dropout(dropout_num ))
        if classes == 1:
            model.add(Dense(classes, activation='sigmoid', name='Output'))
        else:
            model.add(Dense(classes, activation='softmax', name='Output'))
            
        for layer in xception.layers:
            layer.trainable = False
        print("The model summary of Xception  -->\n\n\n")        # In this the Resnet50 layers are not trainable 
        
        for i, layer in enumerate(xception.layers):
            print(i, layer.name, layer.trainable)
        model.compile(loss=loss_param,                # Change according to data
                      optimizer=optimizers.RMSprop(),
                      metrics=['accuracy'])
        print("The summary of final Model \n\n\n")
        print(model.summary())
        print('\n\n\n')
       
       

        fit_history = model.fit_generator(
            train_generator,
            steps_per_epoch=len(train_generator.filenames) // batch,
            epochs=epoch,
            shuffle=True,
            validation_data=validation_generator,
            validation_steps=len(train_generator.filenames) // batch,
            class_weight=n,
            callbacks=[
                EarlyStopping(patience=patience_param, restore_best_weights=True),
                ReduceLROnPlateau(patience=patience_param)
            ])
        os.chdir(output_path)    
        model.save("model.h5")
        print(fit_history.history.keys())
        plt.figure(1, figsize = (15,8)) 

        plt.subplot(221)  
        plt.plot(fit_history.history['accuracy'])  
        plt.plot(fit_history.history['val_accuracy'])  
        plt.title('model accuracy')  
        plt.ylabel('accuracy')  
        plt.xlabel('epoch')  
        plt.legend(['train', 'valid']) 

        plt.subplot(222)  
        plt.plot(fit_history.history['loss'])  
        plt.plot(fit_history.history['val_loss'])  
        plt.title('model loss')  
        plt.ylabel('loss')  
        plt.xlabel('epoch')  
        plt.legend(['train', 'valid']) 

        plt.show()
        
              
    if trainable is not None:
        # Make last block of the conv_base trainable:

        for layer in xception.layers[:trainable]:
            layer.trainable = False
        for layer in xception.layers[trainable:]:
            layer.trainable = True

        print('Last block of the conv_base is now trainable')
        
        for i, layer in enumerate(xception.layers):
            print(i, layer.name, layer.trainable)
            
        model = Sequential()
        model.add(xception)
        model.add(Dense(hidden, activation='relu', input_dim=input_sh))
        model.add(Dropout(dropout_num))
        model.add(Dense(hidden, activation='relu'))
        model.add(Dropout(dropout_num ))
        model.add(Dense(hidden, activation='relu'))
        model.add(Dropout(dropout_num ))
        if classes == 1:
            model.add(Dense(classes, activation='sigmoid', name='Output'))
        else:
            model.add(Dense(classes, activation='softmax', name='Output'))
            
        for layer in xception.layers:
            layer.trainable = False
        print("The model summary of Xception  -->\n\n\n")        # In this the Resnet50 layers are not trainable     
        model.compile(loss=loss_param,                # Change according to data
                      optimizer=optimizers.RMSprop(),
                      metrics=['accuracy'])
        print("The summary of final Model \n\n\n")
        print(model.summary())
        print('\n\n\n')
       
       

        fit_history = model.fit_generator(
            train_generator,
            steps_per_epoch=len(train_generator.filenames) // batch,
            epochs=epoch,
            shuffle=True,
            validation_data=validation_generator,
            validation_steps=len(train_generator.filenames) // batch,
            class_weight=n,
            callbacks=[
                EarlyStopping(patience=patience_param, restore_best_weights=True),
                ReduceLROnPlateau(patience=patience_param)
            ])
        os.chdir(output_path)    
        model.save("model.h5")
        print(fit_history.history.keys())
        plt.figure(1, figsize = (15,8)) 

        plt.subplot(221)  
        plt.plot(fit_history.history['accuracy'])  
        plt.plot(fit_history.history['val_accuracy'])  
        plt.title('model accuracy')  
        plt.ylabel('accuracy')  
        plt.xlabel('epoch')  
        plt.legend(['train', 'valid']) 

        plt.subplot(222)  
        plt.plot(fit_history.history['loss'])  
        plt.plot(fit_history.history['val_loss'])  
        plt.title('model loss')  
        plt.ylabel('loss')  
        plt.xlabel('epoch')  
        plt.legend(['train', 'valid']) 

        plt.show()


# #### Xception_Net()  --> Only for Top layer addition not for fine tuning
# #### Xception_Net(trainable = 120)  -->For fine tuning (Means all layer after 120 is trainable)

# In[ ]:


Xception_Net(trainable = 120)

