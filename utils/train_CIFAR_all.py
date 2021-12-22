
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler,ModelCheckpoint
import matplotlib.pyplot as plt


def cifar_train(model,path,X_train,y_train,X_val,y_val,data_aug=True,learning_rate=0.1,gamma = 0.1, epochs=200, epochs_schedule = [80, 140], batch_size=128):

    if data_aug:
        train_datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        horizontal_flip=True,
        width_shift_range=2, #+-4 pixel shift max
        height_shift_range=2)#+-4 pixel shift max

        train_datagen.fit(X_train)
        
        # Apply the desired normalization
        X_val_std = X_val - train_datagen.mean
        X_val_std /= train_datagen.std
        
    else:
         train_datagen = ImageDataGenerator()
    # Change learning rate
    def change_learning_rate(epoch, lr):
        if epoch in epochs_schedule:
            # print('current learning rate: ',round(lr, 4))
            return lr * gamma
        return lr
    
    lr_callback = [LearningRateScheduler(change_learning_rate, verbose=1)]

    # checkpoints
    checkpoint = ModelCheckpoint(path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate,momentum=0.9),loss='categorical_crossentropy', metrics = ['acc'])


    train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
    step_size_train=train_generator.n//train_generator.batch_size

    history=model.fit(train_generator,
                       steps_per_epoch = step_size_train,
                       epochs = epochs,
                       validation_data=(X_val_std, y_val),
                       callbacks=[lr_callback,checkpoint])
    
    return history

def plot_model(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy', color="blue")
    plt.plot(val_acc, label='Validation Accuracy', color="r")
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss', color="blue")
    plt.plot(val_loss, label='Validation Loss', color="r")
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')

    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

def plot_model_inception(history):
    acc = history.history['prediction_layer_acc']
    val_acc = history.history['val_prediction_layer_acc']

    loss = history.history['prediction_layer_loss']
    val_loss = history.history['val_prediction_layer_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy', color="blue")
    plt.plot(val_acc, label='Validation Accuracy', color="r")
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss', color="blue")
    plt.plot(val_loss, label='Validation Loss', color="r")
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')

    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

