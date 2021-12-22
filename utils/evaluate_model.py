# Evaluate the model
# works for one hot labels

import tensorflow as tf
import numpy as np


##### top_n_acc ######
# pred: list of porbabilisty predictions of the model
# targets: list of ture labels
# n: integer that determines the top n accuracy to compute
#
# output : returns top n accuracy
#######################
def top_n_acc(pred, targets,n=1):
        acc=np.mean(tf.keras.metrics.top_k_categorical_accuracy(targets, pred, k=n).numpy())
        print("top",n,'accuracy:',acc*100,sep=' ')
        return acc
    

###### SCORE #######
# model: tf model
# X testing data
# y testing labels
# top_n: list of integers n, list of top n accuracies of evaluate
#
###################
def score(model,X,y,top_n=[]):
    model.compile(loss='categorical_crossentropy', metrics = ['acc'])
    loss, acc = model.evaluate(X, y, verbose=2)
    print(model._name)
    print("model accuracy: {:5.2f}%".format(100 * acc))
    
    for n in top_n:
        preds=model.predict(X)
        top_n_acc(preds,y,n)


def score_inception(model,X,y,top_n=[]):
    model.compile(loss='categorical_crossentropy', metrics = ['acc'])
    eval = model.evaluate(X, y, verbose=2)
    loss = eval[2]
    acc = eval[4]
    print(model._name)
    print("model accuracy: {:5.2f}%".format(100 * acc))
    
    for n in top_n:
        preds=model.predict(X)[1]
        top_n_acc(preds,y,n)