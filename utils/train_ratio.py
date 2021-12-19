from utils.SE_resnet import SE_custom_resnet
import os
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler,ModelCheckpoint,ReduceLROnPlateau
import matplotlib.pyplot as plt
import tensorflow as tf
from utils.evaluate_model import score

from utils.train_CIFAR_ResNet import plot_model

def train_ratio(ratio,X_train,y_train,X_val,y_val,X_test, y_test):
    SE_resnet=SE_custom_resnet(res_model=[[[3,3],[64,64],4],[[3,3],[128,128],6],[[1,3,1],[256,256,1024],8]],custom_input=(32,32,3),n_classes=10,ratio=ratio,model_name='custom_resnet46_ratio1',debug=False)
    SE_resnet.summary()

    
    path='./models/ratio/se_resnet_ratio_'+str(ratio)+'.hdf5'
    train_datagen = ImageDataGenerator(
            horizontal_flip=True,
            width_shift_range=4, #+-8 pixel shift max
            height_shift_range=4)#+-8 pixel shift max


    reduce_lr = ReduceLROnPlateau(monitor='val_acc', mode='max',factor=0.1,
                                  patience=5, min_lr=0.001, verbose=1)

    #Early stopping
    es = tf.keras.callbacks.EarlyStopping(monitor='val_acc', mode='max', patience=10)
    # checkpoints
    checkpoint = ModelCheckpoint(path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    SE_resnet.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01,momentum=0.9),loss='categorical_crossentropy', metrics = ['acc'])


    train_generator = train_datagen.flow(X_train, y_train, batch_size=128)
    step_size_train=train_generator.n//train_generator.batch_size

    history=SE_resnet.fit(train_generator,
                        steps_per_epoch=step_size_train,
                        epochs=200,
                        validation_data=(X_val,y_val),
                        callbacks=[reduce_lr,es,checkpoint])

    plot_model(history)


    print(history.history['acc'])
    print(history.history['loss'])
    print(history.history['val_acc'])
    print(history.history['val_loss'])


    from utils.evaluate_model import score
    # reload from saved weights:
    model=SE_custom_resnet(res_model=[[[3,3],[64,64],4],[[3,3],[128,128],6],[[1,3,1],[256,256,1024],8]],custom_input=(32,32,3),n_classes=10,ratio=ratio,model_name='custom_resnet46_ratio1',debug=False)
    model.load_weights(path)

    model.save('./full_models/ratio/se_resnet_ratio_'+str(ratio)+'.h5')
    del model  # deletes the existing model
    model = tf.keras.models.load_model('./full_models/ratio/se_resnet_ratio_'+str(ratio)+'.h5')
    score(model,X_test,y_test,top_n=[3,5])