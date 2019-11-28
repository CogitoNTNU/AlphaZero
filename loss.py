import tensorflow as tf
from keras.layers import Activation
from keras.models import Model
import numpy as np


def softmax_cross_entropy_with_logits(y_true, y_pred):
    p = y_pred
    pi = y_true

    zero = tf.zeros(shape=tf.shape(pi), dtype=tf.float32)
    where = tf.equal(pi, zero)

    negatives = tf.fill(tf.shape(pi), -100.0)
    p = tf.where(where, negatives, p)


    loss = tf.nn.softmax_cross_entropy_with_logits(labels=pi, logits=p)

    return loss


def softmax(y_true, y_pred):
    p = y_pred
    pi = y_true

    zero = np.zeros(pi.shape)
    where = np.equal(pi, zero)

    negatives = np.full(pi.shape, -100.0)
    p = np.where(where, negatives, p)

    # print('lin_act', p)
    return np.exp(p - np.max(p)) / np.exp(p - np.max(p)).sum()

