import os
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import f1_score, confusion_matrix
import itertools

# Data preparation
train_datagen = ImageDataGenerator(rescale=1./255)
train_set = train_datagen.flow_from_directory('/Users/vincentiusverel/Vincent/TugasAkhir/TA_EmotionDetection/dataset/fer2013plus/fer2013/train',
                                              target_size=(48, 48),
                                              batch_size=32,
                                              class_mode='binary',
                                              shuffle=True,
                                              color_mode="grayscale")

test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory('/Users/vincentiusverel/Vincent/TugasAkhir/TA_EmotionDetection/dataset/fer2013plus/fer2013/test',
                                            target_size=(48, 48),
                                            batch_size=32,
                                            class_mode='binary',
                                            shuffle=False,
                                            color_mode="grayscale")

# Model definition
basemodel = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='softmax')
])

# Compile model
initial_learning_rate = 0.0001
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
basemodel.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

# Learning rate scheduler
def lr_schedule(epoch):
    if epoch < 20:
        return initial_learning_rate
    else:
        return initial_learning_rate * 0.1

# F1-Score callback
class F1_Score(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        super(F1_Score, self).__init__()
        self.validation_data = validation_data
        self.f1_scores = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict, average='macro')
        self.f1_scores.append(_val_f1)
        logs['val_f1'] = _val_f1
        print(f' — val_f1: {_val_f1:.4f}')

# Callbacks
file_name = 'tesmodel.h5'
checkpoint_path = os.path.join('model', file_name)
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                       monitor='val_accuracy',
                                       verbose=1,
                                       save_best_only=True,
                                       mode='max'),
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10),
    tf.keras.callbacks.LearningRateScheduler(lr_schedule),
    # F1_Score(validation_data=(X_val, y_val))  # Add this callback if you have prepared validation data
]

# Model training
history = basemodel.fit(train_set,
                        epochs=1,
                        validation_data=test_set,
                        callbacks=callbacks)

# Plotting
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')
plt.show()

# Assuming you have the predictions and true labels for the test set
# y_pred = model.predict(test_set)
# y_true = np.array([test_set.labels]).T  # Adjust based on your data
# f1_score_value = f1_score(y_true, y_pred.round(), average='macro')
# print(f"F1 Score: {f1_score_value}")

# Generate confusion matrix
# cm = confusion_matrix(y_true, y_pred.round())
# plot_confusion_matrix(cm, classes=['Class 0', 'Class 1'])  # Adjust classes accordingly
