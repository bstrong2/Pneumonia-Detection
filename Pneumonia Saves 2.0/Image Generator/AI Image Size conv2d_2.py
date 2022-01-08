import os
import time

import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from keras.models import Model, load_model
from keras.layers import Dense, Input, Conv2D, MaxPool2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf

xrayPath = 'C:\\Users\\asus\\Downloads\\chest_xray\\'

train = 'train\\'
test = 'test\\'
val = 'val\\'

bacteria = 'Bacteria'
normal = 'Normal'
virus = 'Virus'

np.random.seed(0)

fileOutput = 'D:\\Pneumonia Saves 2.0\\Image Generator\\'

#
# AI SETTINGS
#
epochs = 120
imageWidth = 250
imageHeight = 250
iteration = 0
totalIterations = 3


def LoadImages(path):
    pathb = path + bacteria
    bacteriaFiles = np.array(os.listdir(pathb))

    pathn = path + normal
    normalFiles = np.array(os.listdir(pathn))

    pathv = path + virus
    virusFiles = np.array(os.listdir(pathv))

    if 'test' in path:
        pathb = pathb + '\\'
        pathn = pathn + '\\'
        pathv = pathv + '\\'

        bacteria_labels = ['Bacteria'] * len(bacteriaFiles)
        bacteria_images = []
        print('Loading test set bacteria photos')
        time.sleep(1)
        for i in tqdm(bacteriaFiles):
            i = cv2.imread(pathb + i)
            i = cv2.resize(i, dsize=(imageWidth, imageHeight))

            # Convert the image to gray scale
            image = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)

            # Make a list of all the modified images
            bacteria_images.append(image)

        # Make the list a numpy array so it's more efficient (and so we can work with it easier)
        #bacteria_images = np.array(bacteria_images)

        normalLabels = ['Normal'] * len(normalFiles)
        normal_images = []
        print('Loading test set normal photos')
        time.sleep(1)
        for i in tqdm(normalFiles):
            i = cv2.imread(pathn + i)
            i = cv2.resize(i, dsize=(imageWidth, imageHeight))

            # Convert the image to gray scale
            image = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)

            # Make a list of all the modified images
            normal_images.append(image)

        # Make the list a numpy array so it's more efficient (and so we can work with it easier)
        #normal_images = np.array(normal_images)

        virusLabels = ['Virus'] * len(virusFiles)
        virus_images = []
        print('Loading test set virus photos')
        time.sleep(1)
        for i in tqdm(virusFiles):
            i = cv2.imread(pathv + i)
            i = cv2.resize(i, dsize=(imageWidth, imageHeight))

            # Convert the image to gray scale
            image = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)

            # Make a list of all the modified images
            virus_images.append(image)

        # Make the list a numpy array so it's more efficient (and so we can work with it easier)
        #virus_images = np.array(virus_images)

        labels = bacteria_labels + normalLabels + virusLabels
        labels = np.array(labels)
        images = bacteria_images + normal_images + virus_images
        images = np.array(images)
        return labels, images
    elif 'train' in path:
        pathb = pathb + '\\'
        pathn = pathn + '\\'
        pathv = pathv + '\\'

        bacteria_labels = ['Bacteria'] * len(bacteriaFiles)
        bacteria_images = []
        print('Loading train set bacteria photos')
        time.sleep(1)
        for i in tqdm(bacteriaFiles):
            i = cv2.imread(pathb + i)
            i = cv2.resize(i, dsize=(imageWidth, imageHeight))

            # Convert the image to gray scale
            image = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)

            # Make a list of all the modified images
            bacteria_images.append(image)

        # Make the list a numpy array so it's more efficient (and so we can work with it easier)
        #bacteria_images = np.array(bacteria_images)

        normalLabels = ['Normal'] * len(normalFiles)
        normal_images = []
        print('Loading train set normal photos')
        time.sleep(1)
        for i in tqdm(normalFiles):
            i = cv2.imread(pathn + i)
            i = cv2.resize(i, dsize=(imageWidth, imageHeight))

            # Convert the image to gray scale
            image = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)

            # Make a list of all the modified images
            normal_images.append(image)

        # Make the list a numpy array so it's more efficient (and so we can work with it easier)
        #normal_images = np.array(normal_images)

        virusLabels = ['Virus'] * len(virusFiles)
        virus_images = []
        print('Loading train set virus photos')
        time.sleep(1)
        for i in tqdm(virusFiles):
            i = cv2.imread(pathv + i)
            i = cv2.resize(i, dsize=(imageWidth, imageHeight))

            # Convert the image to gray scale
            image = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)

            # Make a list of all the modified images
            virus_images.append(image)

        # Make the list a numpy array so it's more efficient (and so we can work with it easier)
        #virus_images = np.array(virus_images)

        labels = bacteria_labels + normalLabels + virusLabels
        labels = np.array(labels)
        images = bacteria_images + normal_images + virus_images
        images = np.array(images)
        return labels, images

    elif 'val' in path:
        pathb = pathb + '\\'
        pathn = pathn + '\\'
        pathv = pathv + '\\'

        bacteria_labels = ['Bacteria'] * len(bacteriaFiles)
        bacteria_images = []
        print('Loading validation set bacteria photos')
        time.sleep(1)
        for i in tqdm(bacteriaFiles):
            i = cv2.imread(pathb + i)
            i = cv2.resize(i, dsize=(imageWidth, imageHeight))

            # Convert the image to gray scale
            image = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)

            # Make a list of all the modified images
            bacteria_images.append(image)

        # Make the list a numpy array so it's more efficient (and so we can work with it easier)
        #bacteria_images = np.array(bacteria_images)

        normalLabels = ['Normal'] * len(normalFiles)
        normal_images = []
        print('Loading validation set normal photos')
        time.sleep(1)
        for i in tqdm(normalFiles):
            i = cv2.imread(pathn + i)
            i = cv2.resize(i, dsize=(imageWidth, imageHeight))

            # Convert the image to gray scale
            image = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)

            # Make a list of all the modified images
            normal_images.append(image)

        # Make the list a numpy array so it's more efficient (and so we can work with it easier)
        #normal_images = np.array(normal_images)

        virusLabels = ['Virus'] * len(virusFiles)
        virus_images = []
        print('Loading validation set virus photos')
        time.sleep(1)
        for i in tqdm(virusFiles):
            i = cv2.imread(pathv + i)
            i = cv2.resize(i, dsize=(imageWidth, imageHeight))

            # Convert the image to gray scale
            image = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)

            # Make a list of all the modified images
            virus_images.append(image)

        # Make the list a numpy array so it's more efficient (and so we can work with it easier)
        #virus_images = np.array(virus_images)

        labels = bacteria_labels + normalLabels + virusLabels
        labels = np.array(labels)
        images = bacteria_images + normal_images + virus_images
        images = np.array(images)
        return labels, images


int = 0


found = False
# CHECK IF THERE IS A PICKLE FILE!!
files = [f for f in os.listdir('.') if os.path.isfile(f)]
for f in files:
    if 'pneumonia_' + str(imageWidth) + 'x' + str(imageHeight) + '.pickle' in f:
        found = True
        break

#
# IF THERE ISN'T A '.PICKLE' FILE THEN UNCOMMENT THESE LINES!!!!!
#

if found == False:
    # load the images into memory start with NORMAL
    ytrain, xtrain = LoadImages(xrayPath + train)

    ytest, xtest = LoadImages(xrayPath + test)

    yval, xval = LoadImages(xrayPath + val)

    with open('pneumonia_' + str(imageWidth) + 'x' + str(imageHeight) + '.pickle', 'wb') as f:
        pickle.dump((ytrain, xtrain, ytest, xtest, yval, xval), f)
else:
    with open('pneumonia_' + str(imageWidth) + 'x' + str(imageHeight) + '.pickle', 'rb') as f:
        (ytrain, xtrain, ytest, xtest, yval, xval) = pickle.load(f)

ytrain = ytrain[:, np.newaxis]
ytest = ytest[:, np.newaxis]
yval = yval[:, np.newaxis]

# Initialize the encoder
one_hot_encoder = OneHotEncoder(sparse=False)

# Change the data to a different format
ytrainHot = one_hot_encoder.fit_transform(ytrain)
ytestHot = one_hot_encoder.transform(ytest)
yvalHot = one_hot_encoder.transform(yval)

# Add a color variable to the x train and test sets
xtrain = xtrain.reshape(xtrain.shape[0], xtrain.shape[1], xtrain.shape[2], 1)
xtest = xtest.reshape(xtest.shape[0], xtest.shape[1], xtest.shape[2], 1)
xval = xval.reshape(xval.shape[0], xval.shape[1], xval.shape[2], 1)
reset = 0


rotation = 0


while rotation < 10:

    heightShift = 0

    while heightShift < 1:

        widthShift = 0

        while widthShift < 1:

            zoomRange = 0

            while zoomRange < 2:

                # Create more training samples by changing the photos. (zoom in, rotate image, etc)
                datagen = ImageDataGenerator(
                    rotation_range=rotation,
                    zoom_range=zoomRange,
                    width_shift_range=widthShift,
                    height_shift_range=heightShift)

                # Calculate any statistics required to actually perform the transforms to your image data
                datagen.fit(xtrain)

                # Configure the batch size and prepare the data generator and get batches of images
                train_gen = datagen.flow(xtrain, ytrainHot, batch_size=32)

                # The numbers that we will use for layer 1's input
                input1 = Input(shape=(xtrain.shape[1], xtrain.shape[2], 1))

                cnn = Conv2D(16, (3, 3), activation='relu', strides=(1, 1),
                             padding='same')(input1)
                cnn = Conv2D(32, (3, 3), activation='relu', strides=(1, 1),
                             padding='same')(cnn)
                cnn = MaxPool2D((2, 2))(cnn)

                cnn = Conv2D(16, (2, 2), activation='relu', strides=(1, 1),
                             padding='same')(cnn)
                cnn = Conv2D(32, (2, 2), activation='relu', strides=(1, 1),
                             padding='same')(cnn)
                cnn = MaxPool2D((2, 2))(cnn)

                #
                #  Neural network look at documentation for 'Flatten', 'Dense'
                # https://stackoverflow.com/questions/43237124/what-is-the-role-of-flatten-in-keras
                # https://keras.io/api/layers/core_layers/dense/
                #
                cnn = Flatten()(cnn)
                cnn = Dense(150, activation='relu')(cnn)
                cnn = Dense(610, activation='relu')(cnn)
                output1 = Dense(3, activation='softmax')(cnn)

                #
                # Model groups layers into an object with training and inference features.
                # https://keras.io/api/models/model/#model-class
                #
                model = Model(inputs=input1, outputs=output1)

                # https://keras.io/api/models/model_training_apis/#compile-method
                model.compile(loss='categorical_crossentropy',
                              optimizer='adam', metrics=['acc'])

                earlyStopping = EarlyStopping(monitor='val_acc', patience=epochs, verbose=0, mode='max')
                mcp_save = ModelCheckpoint(
                    fileOutput + 'Model Saves\\' + 'Model Saves_' + str(imageWidth) + 'x' + str(imageHeight)
                    + '_W' + str(widthShift) + '_H' + str(heightShift) + '_Z' + str(
                        zoomRange)
                    + '_R' + str(rotation) + '.h5', save_best_only=True, monitor='val_acc', mode='max')
                # reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4,
                #                                   mode='min')

                # https://www.geeksforgeeks.org/keras-fit-and-keras-fit_generator/
                history = model.fit_generator(train_gen, epochs=epochs,
                                              validation_data=(xtest, ytestHot),
                                              callbacks=[earlyStopping, mcp_save])

                model = load_model(
                    fileOutput + 'Model Saves\\' + 'Model Saves_' + str(imageWidth) + 'x' + str(imageHeight)
                    + '_W' + str(widthShift) + '_H' + str(heightShift) + '_Z' + str(zoomRange)
                    + '_R' + str(rotation) + '.h5')

                predictions = model.predict(xtest)
                predictions = one_hot_encoder.inverse_transform(predictions)

                # Make the confusion matrix with the predictions vs the actual values
                cm = confusion_matrix(ytest, predictions)

                total = 0

                # Get the total for the confusion matrix
                for i in cm:
                    for j in i:
                        total += j

                # Do the calculations for the confusion matrix
                percent = ((cm[0][0] + cm[1][1] + cm[2][2]) / total) * 100
                sensitivity = (cm[0][0] / (cm[0][0] + cm[0][1] + cm[0][2])) * 100
                specificity = ((cm[1][1] + cm[1][2] + cm[2][1] + cm[2][2]) / (
                            cm[1][0] + cm[1][1] + cm[1][2] + cm[2][0] + cm[2][1] +
                            cm[2][2])) * 100
                ppv = (cm[0][0] / (cm[0][0] + cm[1][0] + cm[2][0])) * 100
                npv = ((cm[1][1] + cm[1][2] + cm[2][1] + cm[2][2]) / (
                        cm[0][1] + cm[0][2] + cm[1][1] + cm[1][2] + cm[2][1] + cm[2][2])) * 100
                accuracyBinary = ((cm[0][0] + cm[1][1] + cm[2][2] + cm[0][2] + cm[2][0]) / total) * 100
                fn = cm[0][1] + cm[2][1]
                precision = ((cm[0][0] + cm[1][1] + cm[2][2]) / ((cm[0][0] + cm[1][1] + cm[2][2]) + fn)) * 100

                # Open the file and write the calculated information into the file
                file = open(fileOutput + 'Confusion Matrix Info.txt', 'a')
                file.write(
                    '\nModel: ' + str(imageWidth) + 'x' + str(imageHeight) + '_W' + str(
                        widthShift) + '_H' + str(heightShift) + '_Z' + str(zoomRange)
                    + '_R' + str(rotation))
                file.write('\nAccuracy: ')
                file.write(str(percent) + '%')
                file.write('\nSensitivity: ')
                file.write(str(sensitivity) + '%')
                file.write('\nSpecificity: ')
                file.write(str(specificity) + '%')
                file.write('\nPositive Predictive Value: : ')
                file.write(str(ppv) + '%')
                file.write('\nNegative Predictive Value: ')
                file.write(str(npv) + '%')
                file.write('\nBinary Accuracy: ')
                file.write(str(accuracyBinary) + '%')
                file.write('\nPrecision rate: ' + str(precision) + '%')
                file.write('\nNum Of False Negatives: ')
                file.write(str(fn) + '\n')
                file.close()

                classnames = ['Bacteria', 'Normal', 'Virus']
                plt.figure(figsize=(8, 8))
                plt.title('Confusion Matrix')
                sns.heatmap(cm, cbar=False, xticklabels=classnames, yticklabels=classnames, fmt='d', annot=True,
                            cmap=plt.cm.Blues)
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.savefig(fileOutput + 'Confusion Matrix\\' + 'Model Saves_' + str(imageWidth) + 'x' + str(
                    imageHeight) + '_W' + str(widthShift) + '_H' + str(heightShift) + '_Z' + str(
                    zoomRange)
                            + '_R' + str(rotation) + '.png', bbox_inches='tight')

                # Plot the first values
                plt.figure(figsize=(8, 6))
                plt.title('Accuracy scores')
                plt.plot(history.history['acc'])
                plt.plot(history.history['val_acc'])
                plt.legend(['acc', 'val_acc'])
                plt.savefig(
                    fileOutput + 'Accuracy\\' + 'Model Saves_' + str(imageWidth) + 'x' + str(imageHeight)
                    + '_W' + str(widthShift) + '_H' + str(heightShift) + '_Z' + str(zoomRange)
                    + '_R' + str(rotation) + '.png', bbox_inches='tight')

                # Plot the second values
                plt.figure(figsize=(8, 6))
                plt.title('Loss value')
                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                plt.legend(['loss', 'val_loss'])
                plt.savefig(fileOutput + 'Loss\\' + 'Model Saves_' + str(imageWidth) + 'x' + str(imageHeight)
                            + '_W' + str(widthShift) + '_H' + str(heightShift) + '_Z' + str(
                    zoomRange)
                            + '_R' + str(rotation) + '.png', bbox_inches='tight')

                # reset += 1
                # if reset % 2 == 0:
                del cnn
                del model
                del output1

                predictions = model.predict(xval)
                predictions = one_hot_encoder.inverse_transform(predictions)

                # Make the confusion matrix with the predictions vs the actual values
                cm = confusion_matrix(ytest, predictions)

                total = 0

                # Get the total for the confusion matrix
                for i in cm:
                    for j in i:
                        total += j

                # Do the calculations for the confusion matrix
                percent = ((cm[0][0] + cm[1][1] + cm[2][2]) / total) * 100
                sensitivity = (cm[0][0] / (cm[0][0] + cm[0][1] + cm[0][2])) * 100
                specificity = ((cm[1][1] + cm[1][2] + cm[2][1] + cm[2][2]) / (
                        cm[1][0] + cm[1][1] + cm[1][2] + cm[2][0] + cm[2][1] +
                        cm[2][2])) * 100
                ppv = (cm[0][0] / (cm[0][0] + cm[1][0] + cm[2][0])) * 100
                npv = ((cm[1][1] + cm[1][2] + cm[2][1] + cm[2][2]) / (
                        cm[0][1] + cm[0][2] + cm[1][1] + cm[1][2] + cm[2][1] + cm[2][2])) * 100
                accuracyBinary = ((cm[0][0] + cm[1][1] + cm[2][2] + cm[0][2] + cm[2][0]) / total) * 100
                fn = cm[0][1] + cm[2][1]
                precision = ((cm[0][0] + cm[1][1] + cm[2][2]) / ((cm[0][0] + cm[1][1] + cm[2][2]) + fn)) * 100

                # Open the file and write the calculated information into the file
                file = open(fileOutput + 'Confusion Matrix Info Validation', 'a')
                file.write(
                    '\nModel: ' + str(imageWidth) + 'x' + str(imageHeight) + '_W' + str(
                        widthShift) + '_H' + str(heightShift) + '_Z' + str(zoomRange)
                    + '_R' + str(rotation))
                file.write('\nAccuracy: ')
                file.write(str(percent) + '%')
                file.write('\nSensitivity: ')
                file.write(str(sensitivity) + '%')
                file.write('\nSpecificity: ')
                file.write(str(specificity) + '%')
                file.write('\nPositive Predictive Value: : ')
                file.write(str(ppv) + '%')
                file.write('\nNegative Predictive Value: ')
                file.write(str(npv) + '%')
                file.write('\nBinary Accuracy: ')
                file.write(str(accuracyBinary) + '%')
                file.write('\nPrecision rate: ' + str(precision) + '%')
                file.write('\nNum Of False Negatives: ')
                file.write(str(fn) + '\n')
                file.close()

                file.close()

                zoomRange += 0.05

            widthShift += 0.05

        heightShift += 0.05

    rotation += 0.05






