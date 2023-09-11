import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

optimizer=keras.optimizers.RMSprop(learning_rate=0.01)

#print('There are {} images in the dataset'.format(len(glob.glob('/Users/abuzanki/Document_GTPRO/BRIN_2023/MonLink2023/data/Co-60/Ft_Image/*.png'))))
#X_train, X_test, y_train, y_test = train_test_split()

Co60_map = []
Cs137_map = []
Cs134_map = []
Eu152_map = []

def insert_data (srcname) :
 file_path = os.getcwd() + '/data/' + srcname + '/Ft_Image/'

 for img in glob.glob(file_path + '*.png'):
    if (srcname == 'Co-60'):
        Co60_map.append(img)
    elif (srcname == 'Cs-137') :
        Cs137_map.append(img)
    elif (srcname == 'Cs-134') :
        Cs134_map.append(img)
    elif (srcname == 'Eu-152') :
        Eu152_map.append(img)

def data_partition () :
 
 #file_path = os.getcwd() + '/data/' + srcname + '/Ft_Image/'

   #shuffle the lists
 np.random.shuffle(Co60_map)
 np.random.shuffle(Cs137_map)
 np.random.shuffle(Cs134_map)
 np.random.shuffle(Eu152_map)

 #split the data into train, validation and test sets
 train_Cs137, val_Cs137, test_Cs137 = np.split(Cs137_map, [int(len(Cs137_map)*0.7), int(len(Cs137_map)*0.8)])
 train_Co60, val_Co60, test_Co60 = np.split(Co60_map, [int(len(Co60_map)*0.7), int(len(Co60_map)*0.8)])
 train_Cs134, val_Cs134, test_Cs134 = np.split(Cs134_map, [int(len(Cs134_map)*0.7), int(len(Cs134_map)*0.8)])
 train_Eu152, val_Eu152, test_Eu152 = np.split(Eu152_map, [int(len(Eu152_map)*0.7), int(len(Eu152_map)*0.8)])

 train_Cs137_df = pd.DataFrame({'image':train_Cs137, 'label':'Cs-137'})
 val_Cs137_df = pd.DataFrame({'image':val_Cs137, 'label':'Cs-137'})
 test_Cs137_df = pd.DataFrame({'image':test_Cs137, 'label':'Cs-137'})

 train_Co60_df = pd.DataFrame({'image':train_Co60, 'label':'Co-60'})
 val_Co60_df = pd.DataFrame({'image':val_Co60, 'label':'Co-60'})
 test_Co60_df = pd.DataFrame({'image':test_Co60, 'label':'Co-60'})

 train_Cs134_df = pd.DataFrame({'image':train_Cs134, 'label':'Cs-134'})
 val_Cs134_df = pd.DataFrame({'image':val_Cs134, 'label':'Cs-134'})
 test_Cs134_df = pd.DataFrame({'image':test_Cs134, 'label':'Cs-134'})

 train_Eu152_df = pd.DataFrame({'image':train_Eu152, 'label':'Eu-152'})
 val_Eu152_df = pd.DataFrame({'image':val_Eu152, 'label':'Eu-152'})
 test_Eu152_df = pd.DataFrame({'image':test_Eu152, 'label':'Eu-152'})

 train_df = pd.concat([train_Cs137_df, train_Co60_df, train_Cs134_df, train_Eu152_df])
 val_df = pd.concat([val_Cs137_df, val_Co60_df, val_Cs134_df, val_Eu152_df])
 test_df = pd.concat([test_Cs137_df, test_Co60_df, test_Cs134_df, test_Eu152_df])

 BATCH_SIZE = 32
 #IMG_HEIGHT = 224
 #IMG_WIDTH = 224

 #create the ImageDataGenerator object and rescale the images
 trainGenerator = ImageDataGenerator()#rescale=1./255.)
 valGenerator = ImageDataGenerator()#rescale=1./255.)
 testGenerator = ImageDataGenerator()#rescale=1./255.)

 #convert them into a dataset
 trainDataset = trainGenerator.flow_from_dataframe(
   dataframe=train_df,
   class_mode="categorical",
   x_col="image",
   y_col="label",
   batch_size=BATCH_SIZE,
   seed=42,
   shuffle=True,
   #target_size=(IMG_HEIGHT,IMG_WIDTH) #set the height and width of the images
 )

 valDataset = valGenerator.flow_from_dataframe(
   dataframe=val_df,
   class_mode="categorical",
   x_col="image",
   y_col="label",
   batch_size=BATCH_SIZE,
   seed=42,
   shuffle=True,
   #target_size=(IMG_HEIGHT,IMG_WIDTH)
 )

 testDataset = testGenerator.flow_from_dataframe(
   dataframe=test_df,
   class_mode="categorical",
   x_col="image",
   y_col="label",
   batch_size=BATCH_SIZE,
   seed=42,
   shuffle=True,
   #target_size=(IMG_HEIGHT,IMG_WIDTH)
 )

 images, labels = next(iter(testDataset))
 #print(images)
 #print(labels)

 num_classes = len(testDataset.class_indices)

 model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(300, 300, 3)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(256, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(512, (3, 3), activation='relu'),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(num_classes, activation='sigmoid')
 ])

 #return model, testDataset, trainDataset, valDataset

#def evaluate_model(model, testDataset, trainDataset, valDataset) :
 epochs=15

 model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.legacy.RMSprop(learning_rate=1e-4), metrics=['accuracy'])
 history = model.fit(trainDataset, epochs=epochs, validation_data=(valDataset))

 plt.plot(history.history['accuracy'])
 plt.plot(history.history['val_accuracy'])
 plt.xlabel('Epoch')
 plt.ylabel('Accuracy')
 plt.legend(['Training', 'Validation'])
 plt.show()

 loss, acc = model.evaluate(testDataset)
 
 file_path = os.getcwd()
 model.save(file_path + '/model_save/model.keras')
 model.save_weights(file_path + '/model_save/model.h5')

 print('Loss:', loss)
 print('Accuracy:', acc)

#def conf_matrix (testDataset) :
 #file_path = os.getcwd()
 #Lastmodel = keras.models.load_model(file_path + '/model_save/model.keras', compile=True)
 Y_pred = model.predict(testDataset)
 Y_pred_classes = np.argmax(Y_pred, axis=1)
 Y_true = testDataset.class_indices #np.argmax(test_df, axis = 1)
 confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
 f, ax = plt.subplots(figsize = (8, 8))
 sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)
 plt.xlabel("Predicted Label")
 plt.ylabel("True Label")
 plt.title("Confusion Matrix")
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