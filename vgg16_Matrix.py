import pandas as pd
import glob
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay
import random

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from PIL import Image

# Prepare CNN Model: AlexNet
from keras import Model, Input, regularizers
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout
from keras.layers import BatchNormalization
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten, ZeroPadding2D

import datetime

optimizer=keras.optimizers.legacy.RMSprop(learning_rate=0.01)

#print('There are {} images in the dataset'.format(len(glob.glob('/Users/abuzanki/Document_GTPRO/BRIN_2023/MonLink2023/data/Co-60/Ft_Image/*.png'))))
#X_train, X_test, y_train, y_test = train_test_split()

Co60_map = []
Cs137_map = []
Cs134_map = []
Eu152_map = []

def insert_data (srcname) :
 file_path = os.getcwd() + '/data/' + srcname + '/Ft_Extract/'

 for img in glob.glob(file_path + '*.csv'):
    if (srcname == 'Co-60'):
        Co_data = pd.read_csv(img, header=None)
        Co_arr = np.array(Co_data).flatten()
        Co_arr = Co_arr.astype(np.uint8)
        Co_arr = np.append(Co_arr, 0)
        Co60_map.append(Co_arr)
    elif (srcname == 'Cs-134') :
        Cs4_data = pd.read_csv(img, header=None)
        Cs4_arr = np.array(Cs4_data).flatten()
        Cs4_arr = Cs4_arr.astype(np.uint8)
        Cs4_arr = np.append(Cs4_arr, 1)
        Cs134_map.append(Cs4_arr)
    elif (srcname == 'Cs-137') :
        Cs7_data = pd.read_csv(img, header=None)
        Cs7_arr = np.array(Cs7_data).flatten()
        Cs7_arr = Cs7_arr.astype(np.uint8)
        Cs7_arr = np.append(Cs7_arr, 2)
        Cs137_map.append(Cs7_arr)
    elif (srcname == 'Eu-152') :
        Eu_data = pd.read_csv(img, header=None)
        Eu_arr = np.array(Eu_data).flatten()
        Eu_arr = Eu_arr.astype(np.uint8)
        Eu_arr = np.append(Eu_arr, 3)
        Eu152_map.append(Eu_arr)

def data_partition () :
 
 #file_path = os.getcwd() + '/data/' + srcname + '/Ft_Image/'

   #shuffle the lists
 np.random.shuffle(Co60_map)
 np.random.shuffle(Cs137_map)
 np.random.shuffle(Cs134_map)
 np.random.shuffle(Eu152_map)

 #split the data into train, validation and test sets
 train_Cs137, val_Cs137, test_Cs137 = np.split(Cs137_map, [int(len(Cs137_map)*0.6), int(len(Cs137_map)*0.9)])
 train_Co60, val_Co60, test_Co60 = np.split(Co60_map, [int(len(Co60_map)*0.6), int(len(Co60_map)*0.9)])
 train_Cs134, val_Cs134, test_Cs134 = np.split(Cs134_map, [int(len(Cs134_map)*0.6), int(len(Cs134_map)*0.9)])
 train_Eu152, val_Eu152, test_Eu152 = np.split(Eu152_map, [int(len(Eu152_map)*0.6), int(len(Eu152_map)*0.9)])

 kolom = []
 for i in range(1024) :
  kolom.append('pixel'+str(i))
  if i == 1023 :
   kolom.append('label')

 train_Cs137_df = pd.DataFrame(train_Cs137, columns= kolom)
 val_Cs137_df = pd.DataFrame(val_Cs137, columns=kolom)
 test_Cs137_df = pd.DataFrame(test_Cs137, columns=kolom)
 
 train_Co60_df = pd.DataFrame(train_Co60, columns= kolom)
 val_Co60_df = pd.DataFrame(val_Co60, columns=kolom)
 test_Co60_df = pd.DataFrame(test_Co60, columns=kolom)

 train_Cs134_df = pd.DataFrame(train_Cs134, columns= kolom)
 val_Cs134_df = pd.DataFrame(val_Cs134, columns= kolom)
 test_Cs134_df = pd.DataFrame(test_Cs134, columns= kolom)

 train_Eu152_df = pd.DataFrame(train_Eu152, columns= kolom)
 val_Eu152_df = pd.DataFrame(val_Eu152, columns= kolom)
 test_Eu152_df = pd.DataFrame(test_Eu152, columns= kolom)

 train_df = pd.concat([train_Cs137_df, train_Co60_df, train_Cs134_df, train_Eu152_df])
 val_df = pd.concat([val_Cs137_df, val_Co60_df, val_Cs134_df, val_Eu152_df])
 test_df = pd.concat([test_Cs137_df, test_Co60_df, test_Cs134_df, test_Eu152_df])

 train_df = train_df.sample(frac=1)
 val_df = val_df.sample(frac=1)
 test_df = test_df.sample(frac=1)

 train_label = np.array(train_df.label)
 val_label = np.array(val_df.label)
 test_label = np.array(test_df.label)

 train_df.drop('label',axis=1, inplace=True)
 val_df.drop('label',axis=1, inplace=True)
 test_df.drop('label',axis=1, inplace=True)

 train_data = np.array(train_df).reshape(train_df.shape[0],32,32,1)
 val_data = np.array(val_df).reshape(val_df.shape[0],32,32,1)
 test_data = np.array(test_df).reshape(test_df.shape[0],32,32,1)

 label_names = ['Co-60', 'Cs-134', 'Cs-137', 'Eu-152']
 
 plt.figure(figsize=(15,7))
 for i in range(12):
    ax=plt.subplot(3,4,i+1)
    rand_index=random.choice(range(len(train_data)))
    plt.imshow(train_data[rand_index],  cmap=plt.cm.binary)
    plt.axis(False)
    plt.title(label_names[int(train_label[rand_index])])
  
 plt.show()

 #BATCH_SIZE = 32
 #IMG_HEIGHT = 224
 #IMG_WIDTH = 224

 #create the ImageDataGenerator object and rescale the images
#  trainGenerator = ImageDataGenerator()#rescale=1./255.)
#  valGenerator = ImageDataGenerator()#rescale=1./255.)
#  testGenerator = ImageDataGenerator()#rescale=1./255.)

 #convert them into a dataset
#  trainDataset = trainGenerator.flow_from_dataframe(
#    dataframe=train_df,
#    class_mode="categorical",
#    x_col="image",
#    y_col="label",
#    batch_size=BATCH_SIZE,
#    seed=42,
#    shuffle=True,
#    #target_size=(IMG_HEIGHT,IMG_WIDTH) #set the height and width of the images
#  )

#  valDataset = valGenerator.flow_from_dataframe(
#    dataframe=val_df,
#    class_mode="categorical",
#    x_col="image",
#    y_col="label",
#    batch_size=BATCH_SIZE,
#    seed=42,
#    shuffle=True,
#    #target_size=(IMG_HEIGHT,IMG_WIDTH)
#  )

#  testDataset = testGenerator.flow_from_dataframe(
#    dataframe=test_df,
#    class_mode="categorical",
#    x_col="image",
#    y_col="label",
#    batch_size=BATCH_SIZE,
#    seed=42,
#    shuffle=True,
#    #target_size=(IMG_HEIGHT,IMG_WIDTH)
#  )

#  images, labels = next(iter(testDataset))
#  #print(images)
#  #print(labels)

#  num_classes = len(testDataset.class_indices)
#  print(testDataset.class_indices)
 
 #Set random seed
 tf.random.set_seed(42)

 # =====================================================================================================
 # Model CNN from Internet
 # model = tf.keras.Sequential([  
 #    tf.keras.layers.Conv2D(filters=5, kernel_size=3, strides=1, padding="same", activation="relu",
 #                           input_shape=(32,32,1)), keras.layers.Conv2D(10,3, padding="valid", activation='relu'),
 #    tf.keras.layers.MaxPool2D(pool_size=2),
 #    tf.keras.layers.Conv2D(15,3, padding="valid", activation='relu'),
 #    tf.keras.layers.Conv2D(20,3, padding="valid", activation='relu'),
 #    tf.keras.layers.MaxPool2D(pool_size=2),
 #    tf.keras.layers.Conv2D(25,3, padding="valid", activation='relu'),
 #    tf.keras.layers.MaxPool2D(pool_size=2),
 #    tf.keras.layers.Flatten(),
 #    tf.keras.layers.Dense(10, activation="softmax")
 # ])
 # =====================================================================================================

 # VGG16 from Firli
 model = Sequential()
 model.add(Conv2D(input_shape=(32,32,1),filters=16,kernel_size=(5,5),strides=(3,3),padding="same", activation="relu"))
 #model.add(Conv2D(filters=16,kernel_size=(5,5),strides=(3,3),padding="same", activation="relu"))
 model.add(MaxPool2D(pool_size=(2,2),strides=(3,3)))
 #model.add(Conv2D(filters=32, kernel_size=(3, 3),strides=(1,1), padding="same", activation="relu"))
 model.add(Conv2D(filters=32, kernel_size=(3, 3),strides=(1,1), padding="same", activation="relu"))
 model.add(MaxPool2D(pool_size=(3,3),strides=(1,1)))
 model.add((ZeroPadding2D(padding=(1,1))))
 #model.add(Conv2D(filters=64, kernel_size=(3,3),strides=(1,1), padding="same", activation="relu"))
# model.add(Conv2D(filters=64, kernel_size=(3,3),strides=(1,1), padding="same", activation="relu"))
 model.add(Conv2D(filters=64, kernel_size=(3,3),strides=(1,1), padding="same", activation="relu"))
 model.add(Flatten())
 model.add(Dense(units=256,activation="relu"))
 model.add(Dense(units=64,activation="relu"))
 model.add(Dense(units=10, activation="softmax"))

 #return model, testDataset, trainDataset, valDataset

 #def evaluate_model(model, testDataset, trainDataset, valDataset) :
 epochs=100
 
 # =====================================================================================================
 # Training optimizer and fitting from unknown Source
 #model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.legacy.RMSprop(learning_rate=1e-4), metrics=['accuracy'])
 #model.compile(loss="sparse_categorical_crossentropy",
 #              optimizer=tf.keras.optimizers.legacy.RMSprop(learning_rate=1e-4),#tf.keras.optimizers.Adam(learning_rate=0.001),
 #              metrics=["accuracy"])
 #history = model.fit(train_data, train_label, epochs=epochs, batch_size=32, validation_data=(val_data, val_label), verbose=1)

 #  plt.plot(history.history['accuracy'])
 #  plt.plot(history.history['val_accuracy'])
 #  plt.xlabel('Epoch')
 #  plt.ylabel('Accuracy')
 #  plt.legend(['Training', 'Validation'])
 #  plt.show()
 # =====================================================================================================
 # Training optimizer and fitting from Firli
 opt = keras.optimizers.Adam(learning_rate=1e-4)
 model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy']) 
 plateau = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr = 1e-6)

 print("Fit model on training data - Vgg16")
 startTime = datetime.datetime.now()

 history = model.fit(
     train_data,
     train_label,
     batch_size = 32,
     epochs = 100,
     validation_data = (val_data, val_label),
     callbacks = [plateau],
     verbose = 1,
     shuffle=True
 )

 endTime = datetime.datetime.now()

 # Compute time consume in training epochs
 diffTime = endTime - startTime
 diffTime = diffTime.total_seconds()/60

 print('\n' + 'Training finished in : ' + str(diffTime) + ' minutes\n')

 plt.plot(history.history['accuracy'])
 plt.plot(history.history['val_accuracy'])
 plt.xlabel('Epoch')
 plt.ylabel('Accuracy')
 plt.legend(['Training', 'Validation'])
 plt.show()


 pd.DataFrame(history.history).plot(title="Train and validation results",figsize=(10,7))
 plt.show()

 loss, acc = model.evaluate(x=test_data, y=test_label, batch_size=32, steps=50)
 
 file_path = os.getcwd()
 model.save(file_path + '/model_save/model.keras')
 model.save_weights(file_path + '/model_save/model.h5')

 print('Loss:', loss)
 print('Accuracy:', acc)

#def conf_matrix (testDataset) :
 #file_path = os.getcwd()
 #Lastmodel = keras.models.load_model(file_path + '/model_save/model.keras', compile=True)
 Y_pred = model.predict(test_data)
 Y_pred_classes = Y_pred.argmax(axis=1)
 Y_true = test_label
 print("Accuracy:", accuracy_score(Y_true, Y_pred_classes)) 
 print("Precision:", precision_score(Y_true, Y_pred_classes, average="weighted"))
 print('Recall:', recall_score(Y_true, Y_pred_classes, average="weighted")) 
 
 confusion_mtx = confusion_matrix(y_true = Y_true, y_pred= Y_pred_classes)
#  f, ax = plt.subplots(figsize = (10, 10))
#  sns.heatmap(confusion_mtx, aspect="auto")#annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)
#  plt.xlabel("Predicted Label")
#  plt.ylabel("True Label")
#  plt.title("Confusion Matrix")
#  plt.show()
 disp= ConfusionMatrixDisplay(confusion_matrix=confusion_mtx,display_labels=label_names)
 fig, ax = plt.subplots(figsize=(10,10))
 disp.plot(ax=ax)
 plt.show()

def main() :
   insert_data('Co-60')
   insert_data('Cs-137')
   insert_data('Cs-134')
   insert_data('Eu-152')
   data_partition()
   #model, testDataset, trainDataset, valDataset = data_partition()
   #evaluate_model(model, testDataset, trainDataset, valDataset)
   #conf_matrix(testDataset)


main()