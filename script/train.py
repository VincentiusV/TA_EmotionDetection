import os
import pandas as pd 
import numpy as np
import tensorflow as tf 
from matplotlib import pyplot 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from sklearn.metrics import f1_score

# Set seed for reproducibility
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

train_datagen = ImageDataGenerator(rescale = 1./255)
train_set = train_datagen.flow_from_directory('C:/Tugas Akhir/TA_EmotionDetection/dataset/fer2013+/fer2013plus/fer2013/train',
                                                    target_size = (48, 48),
                                                    batch_size = 32,
                                                    class_mode = 'binary',
                                                    shuffle = False)

test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('C:/Tugas Akhir/TA_EmotionDetection/dataset/fer2013+/fer2013plus/fer2013/test',
                                                    target_size = (48, 48),
                                                    batch_size = 32,
                                                    class_mode = 'binary',
                                                    shuffle = False)

basemodel = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(30, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(60, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed_value)),
    tf.keras.layers.Dropout(0.5, seed=seed_value),
    tf.keras.layers.Dense(8, activation='softmax', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed_value))
])

initial_learning_rate = 0.0001 

optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
basemodel.compile(optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

def lr_schedule(epoch):
    if epoch < 20:
        return initial_learning_rate
    else:
        return initial_learning_rate * 0.1
    
file_name = '50_2.h5'
    
checkpoint_path= os.path.join('model',file_name)

# class F1_Score(tf.keras.callbacks.Callback):
#     def __init__(self, test_data):
#         super(F1_Score, self).__init__()
#         self.test_data = test_data
#         self.f1_scores = []

#     def on_epoch_end(self, epoch, logs={}):
#         y_true = []
#         y_pred = []
#         for _, labels in self.test_data:
#             y_true.extend(labels)
#             y_pred.extend(np.argmax(self.model.predict(_), axis=1))
#         f1 = f1_score(y_true, y_pred, average='weighted')
#         self.f1_scores.append(f1)
#         print(f' - val_f1_score: {f1:.4f}')
#         logs['val_f1_score'] = f1

basemodel.save_weights('initial_weights.h5')

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

basemodel.load_weights('initial_weights.h5')

history = basemodel.fit(train_set,
                        epochs=50,
                        batch_size=8,
                        validation_data=test_set,
                        callbacks=call_back)


# F1-Score plot per epoch
# pyplot.plot(history.history['val_f1_score'], label='val_f1_score')
# pyplot.xlabel('Epoch')
# pyplot.ylabel('F1 Score')
# pyplot.legend()
# pyplot.show()

# # Plot loss per iteration
# import matplotlib.pyplot as plt
# plt.plot(history.history['loss'], label='loss')
# plt.plot(history.history['val_loss'], label='val_loss')
# plt.legend()

# # Plot accuracy per iteration
# plt.plot(history.history['accuracy'], label='acc')
# plt.plot(history.history['val_accuracy'], label='val_acc')
# plt.legend()

# # Plot confusion matrix
# from sklearn.metrics import confusion_matrix
# import itertools

# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')

#     print(cm)

#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)

#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                 horizontalalignment="center",
#                 color="white" if cm[i, j] > thresh else "black")

#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.show()


# p_test = basemodel.predict(X_test).argmax(axis=1)
# cm = confusion_matrix(y_test, p_test)
# plot_confusion_matrix(cm, list(range(9)))