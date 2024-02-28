import os
import pandas as pd 
import numpy as np
import tensorflow as tf 
from matplotlib import pyplot 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from sklearn.metrics import f1_score
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Training Parameters')
    parser.add_argument('--epochs', type=int, default=10, help='Nmber of epochs while training', required=True)
    parser.add_argument('--bs', type=int, default=32, help='Batch size while training', required=True)
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate', required=True)
    parser.add_argument('--file_name', type=str, default='Train.h5', help='Name of the model file', required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments()
    epochs = args.epochs
    batch_size = args.bs
    learning_rate = args.lr
    file_name = args.file_name

train_datagen = ImageDataGenerator(rescale = 1./255)
train_set = train_datagen.flow_from_directory('/Users/vincentiusverel/Vincent/TugasAkhir/TA_EmotionDetection/dataset/fer2013plus/fer2013/train',
                                                    target_size = (48, 48),
                                                    batch_size = 32,
                                                    class_mode = 'binary',
                                                    shuffle = False,
                                                    color_mode="grayscale")

test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('/Users/vincentiusverel/Vincent/TugasAkhir/TA_EmotionDetection/dataset/fer2013plus/fer2013/test',
                                                    target_size = (48, 48),
                                                    batch_size = 32,
                                                    class_mode = 'binary',
                                                    shuffle = False,
                                                    color_mode="grayscale")

basemodel = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(8, activation='softmax')
])

initial_learning_rate = learning_rate

optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
basemodel.compile(optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

def lr_schedule(epoch):
    if epoch < 20:
        return initial_learning_rate
    else:
        return initial_learning_rate * 0.1
    
file_name = file_name

checkpoint_path= os.path.join('model',file_name)

call_back = [
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                        monitor='val_accuracy',
                                        verbose=1,
                                        save_freq='epoch',
                                        save_best_only=True,
                                        save_weights_only=False,
                                        mode='max'),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=20,
        restore_best_weights=True
    ),
    tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    # F1_Score(test_data=test_set)
]

history = basemodel.fit(train_set,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=test_set,
                        callbacks=call_back)

# # Plot loss per iteration
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()

# # Plot accuracy per iteration
plt.plot(history.history['accuracy'], label='acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
