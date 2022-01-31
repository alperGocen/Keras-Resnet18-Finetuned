# for keras
from classification_models.keras import Classifiers
import tensorflow as tf 
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, ELU, MaxPool2D, Dropout
from tensorflow.keras import initializers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow import keras
from matplotlib import pyplot as plt
from classification_models.keras import Classifiers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import Model

directory = '../../all_enh'

# for tensorflow.keras
# from classification_models.tfkeras import Classifiers
ResNet18, preprocess_input = Classifiers.get('resnet18')

datagen = ImageDataGenerator(rescale=1./255,
    rotation_range=15,
    horizontal_flip=False,
    featurewise_center=True,
    featurewise_std_normalization=True,
    preprocessing_function=preprocess_input,
    validation_split=0.2) # set validation split

train_generator = datagen.flow_from_directory(
    directory,  
    target_size=(224,224),
    batch_size=64,
    class_mode='categorical',
    shuffle=True,
    subset='training') # set as training data

validation_generator = datagen.flow_from_directory(
    directory, # same directory as training data
    target_size=(224,224),
    batch_size=64,
    shuffle=False,
    class_mode='categorical',
    subset='validation') # set as validation data

# build model
base_model = ResNet18(input_shape=(224,224,3), weights='imagenet', include_top=False)
x = keras.layers.Flatten()(base_model.output)
x = keras.layers.Dense(128, activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01))(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dropout(0.3)(x)
x = keras.layers.Dense(320, activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01))(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dropout(0.3)(x)
output = keras.layers.Dense(1776, activation='softmax')(x)
model = keras.models.Model(inputs=[base_model.input], outputs=[output])


model.compile(loss = 'categorical_crossentropy', optimizer= keras.optimizers.Adam(learning_rate=0.001), metrics=['acc'])

# VGG16
number_of_epochs = 400
resnet50_filepath = '/home/alper/Desktop/resnet_models/resnet_50_'+'-saved-model-{epoch:02d}-acc-{val_acc:.2f}.hdf5'
res_checkpoint = tf.keras.callbacks.ModelCheckpoint(resnet50_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
resnet_early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100)
history = model.fit(train_generator, epochs = number_of_epochs , validation_data = validation_generator,callbacks=[res_checkpoint,resnet_early_stopping],verbose=1)


