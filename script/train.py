import os
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback, CSVLogger, ReduceLROnPlateau
from sklearn.metrics import precision_score,recall_score, f1_score, confusion_matrix

def create_custom_model_1():
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(8, activation='softmax')
    ])
    return model

def create_custom_model_2():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Dropout(0.02),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Dropout(0.5), #0.5
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2), #0.2
        tf.keras.layers.Dense(8, activation='softmax')
    ])
    return model

def create_custom_model_3():
    model = tf.keras.models.Sequential([
        # First block
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(48, 48, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        # Four blocks
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.3),
        
        # Downsampling
        tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.4),
        
        # Classification block
        tf.keras.layers.Conv2D(64, (1, 1), activation='relu'),
        tf.keras.layers.Conv2D(8, (1, 1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(8, activation='softmax')
    ])
    
    return model

def create_vgg16_model():
    base_model = tf.keras.applications.VGG16(
        weights='imagenet',  # Load weights pre-trained on ImageNet.
        input_shape=(48, 48, 3),  # Shape of input image.
        include_top=False)  # Do not include the ImageNet classifier at the top.

    # Freeze the base model
    base_model.trainable = False

    # Create new model on top
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(8, activation='softmax')
    ])
    return model

def create_vgg16_model_trainable():
    base_model = tf.keras.applications.VGG16(
        weights='imagenet',  # Load weights pre-trained on ImageNet.
        input_shape=(48, 48, 3),  # Shape of input image.
        include_top=False)  # Do not include the ImageNet classifier at the top.

    # Freeze the base model
    base_model.trainable = True

    # Create new model on top
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(8, activation='softmax')
    ])
    return model

def create_resnet_model():
    base_model = tf.keras.applications.ResNet50(
        weights='imagenet',  # Load weights pre-trained on ImageNet.
        input_shape=(48, 48, 3),  # Shape of input image.
        include_top=False)  # Do not include the ImageNet classifier at the top.

    # Freeze the base model
    base_model.trainable = False

    # Create new model on top
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(8, activation='softmax')
    ])
    return model

def create_resnet_model_trainable():
    base_model = tf.keras.applications.ResNet50(
        weights='imagenet',  # Load weights pre-trained on ImageNet.
        input_shape=(48, 48, 3),  # Shape of input image.
        include_top=False)  # Do not include the ImageNet classifier at the top.

    # Freeze the base model
    base_model.trainable = True

    # Create new model on top
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(8, activation='softmax')
    ])
    return model

def create_inceptionv3_model():
    base_model = tf.keras.applications.InceptionV3(
        weights='imagenet',  # Load weights pre-trained on ImageNet.
        input_shape=(75, 75, 3),  # Shape of input image.
        include_top=False)  # Do not include the ImageNet classifier at the top.

    # Freeze the base model
    base_model.trainable = False

    # Create new model on top
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(8, activation='softmax')
    ])
    return model

def create_inceptionv3_model_trainable():
    base_model = tf.keras.applications.InceptionV3(
        weights='imagenet',  # Load weights pre-trained on ImageNet.
        input_shape=(75, 75, 3),  # Shape of input image.
        include_top=False)  # Do not include the ImageNet classifier at the top.

    # Freeze the base model
    base_model.trainable = True

    # Create new model on top
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(8, activation='softmax')
    ])
    return model

class MetricsCallback(Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_predict = np.argmax(self.model.predict(self.validation_data), axis=-1)
        val_targ = self.validation_data.classes
        _val_precision = precision_score(val_targ, val_predict, average='macro', zero_division=1)
        _val_recall = recall_score(val_targ, val_predict, average='macro', zero_division=1)
        _val_f1 = f1_score(val_targ, val_predict, average='macro', zero_division=1)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        logs['val_precision'] = _val_precision
        logs['val_recall'] = _val_recall
        logs['val_f1'] = _val_f1
        print(f' — val_precision: {_val_precision} — val_recall: {_val_recall} — val_f1: {_val_f1}')
        super().on_epoch_end(epoch, logs)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Training Parameters')
    parser.add_argument('-dataset', type=str, default='ori', help='Name of the dataset to use (ori, augmented, masked)', required=True)
    parser.add_argument('-model', type=str, default='custom1', help='Name of the model to train (vgg16, resnet, inceptionv3, custom1, custom2)', required=True)
    parser.add_argument('-trainable', type=str, default='False', help='For VGG16, InceptionV3, and Resnet previous layer can be trainable or not (True or False)', required=False)
    parser.add_argument('-epochs', type=int, default=10, help='Nmber of epochs while training', required=True)
    parser.add_argument('-bs', type=int, default=32, help='Batch size while training', required=True)
    parser.add_argument('-lr', type=float, default=0.0001, help='Initial learning rate', required=True)
    parser.add_argument('-fn', type=str, default='Train.h5', help='Name of the model file', required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments()
    model_type = args.model
    trainable = args.trainable
    dataset = args.dataset
    epochs = args.epochs
    batch_size = args.bs
    learning_rate = args.lr
    file_name = args.fn

    if dataset == 'ori':
        train_dir = '../dataset/fer2013plus/train/'
    elif dataset == 'augmented':
        train_dir = '../dataset/augmented'
    elif dataset == 'masked':
        train_dir = '../dataset/augmented_masked'
    else:
        raise ValueError("Invalid dataset type. Choose one of 'ori', 'augmented', 'masked'.")
    
    if model_type == 'custom1':
        model = create_custom_model_1()
        color_mode = 'grayscale'
        target_size = (48, 48)
    elif model_type == 'custom2':
        model = create_custom_model_2()
        color_mode = 'grayscale'
        target_size = (48, 48)
    elif model_type == 'custom3':
        model = create_custom_model_3()
        color_mode = 'grayscale'
        target_size = (48, 48)
    elif model_type == 'vgg16':
        if trainable == 'False':
            model = create_vgg16_model()
            color_mode = 'rgb'
            target_size = (48, 48)
        else:
            model = create_vgg16_model_trainable()
            color_mode = 'rgb'
            target_size = (48, 48)
    elif model_type == 'resnet':
        if trainable == 'False':
            model = create_resnet_model()
            color_mode = 'rgb'
            target_size = (48, 48)
        else:
            model = create_resnet_model_trainable()
            color_mode = 'rgb'
            target_size = (48, 48)
    elif model_type == 'inceptionv3':
        if trainable == 'False':
            model = create_inceptionv3_model()
            color_mode = 'rgb'
            target_size = (75, 75)
        else:
            model = create_inceptionv3_model_trainable()
            color_mode = 'rgb'
            target_size = (75, 75)
    else:
        raise ValueError("Invalid model type. Choose one of 'custom', 'vgg16', 'resnet', 'inceptionv3'.")

    train_datagen = ImageDataGenerator(rescale = 1./255)
    train_set = train_datagen.flow_from_directory(train_dir,
                                                    target_size = target_size,
                                                    batch_size = 32,
                                                    class_mode = 'binary',
                                                    shuffle = False,
                                                    color_mode=color_mode)

    test_datagen = ImageDataGenerator(rescale = 1./255)
    test_set = test_datagen.flow_from_directory('../dataset/fer2013plus/test',
                                                        target_size = target_size,
                                                        batch_size = 32,
                                                        class_mode = 'binary',
                                                        shuffle = False,
                                                        color_mode=color_mode)

    initial_learning_rate = learning_rate
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',  # Monitor the validation loss
        factor=0.1,          # Reduce the learning rate by a factor of 0.1
        patience=10,          # Number of epochs with no improvement after which learning rate will be reduced
        verbose=1,           # Int to print a message when the callback takes an action
        mode='auto',         # 'auto' will infer from the direction of the monitored quantity (min or max)
        min_delta=0.0001,    # Minimum change to qualify as an improvement
        cooldown=0,          # Number of epochs to wait before resuming normal operation after lr has been reduced
        min_lr=0             # Lower bound on the learning rate
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
    model.compile(optimizer=optimizer,
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    folder_name = file_name
    model_name = file_name + '.h5'

    checkpoint_path= os.path.join(f'../training/{file_name}',model_name)

    train_log_dir = f'../training/{file_name}'
    log_file = os.path.join(train_log_dir, 'training.log')

    metrics_callback = MetricsCallback(validation_data=test_set)
    csv_logger = CSVLogger(log_file)

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
        reduce_lr,
        metrics_callback,
        csv_logger
    ]

    history = model.fit(train_set,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=test_set,
                            callbacks=call_back)

    hyperparameters = {
        'model_name': file_name,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
    }

    with open(f'../training/{file_name}/hyperparameters.json', 'w') as f:
        json.dump(hyperparameters, f, indent=4)

    os.makedirs(f'../training/{file_name}/fig', exist_ok=True)

    # Plot loss per iteration
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')  # Adds a title
    plt.xlabel('Epochs')  # Adds an x-axis label
    plt.ylabel('Loss')  # Adds a y-axis label
    plt.legend()  # Displays the legend
    plt.savefig(f'../training/{file_name}/fig/loss.png')  # Saves the plot
    plt.clf()  # Clears the current figure

    # Plot accuracy per iteration
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')  # Adds a title
    plt.xlabel('Epochs')  # Adds an x-axis label
    plt.ylabel('Accuracy')  # Adds a y-axis label
    plt.legend()  # Displays the legend
    plt.savefig(f'../training/{file_name}/fig/accuracy.png')  # Saves the plot
    plt.clf()  # Clears the current figure after saving the plot

    # After training is complete
    val_predict = np.argmax(model.predict(test_set), axis=-1)  # Get last epoch predictions
    val_targ = test_set.classes
    cm = confusion_matrix(val_targ, val_predict)

    class_names = ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']

    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.savefig(f'../training/{file_name}/fig/confusion_matrix.png')
    plt.close()