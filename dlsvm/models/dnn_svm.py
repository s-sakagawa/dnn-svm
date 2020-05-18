import os
import sys
import time
from typing import Union, Callable

import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk


class DnnSvm(tfk.Model):
    def __init__(self, model: tfk.Model, penalty, n_class):
        super().__init__()
        self.model = model
        self.C = penalty
        self.n_class = n_class

    @tf.function
    def call(self, inputs, **kwargs):
        return self.model(inputs)

    def loss_l2_svm(self, y_true, y_pred):
        matrix_shape = [tf.shape(y_true)[0], self.n_class]
        weight = self.model.weights[2]

        regularization_loss = 0.5 * tf.reduce_mean(tf.square(weight))
        hinge_loss = tf.reduce_mean(
            tf.square(
                tf.maximum(
                    tf.ones(matrix_shape) - y_pred * y_true, tf.zeros(matrix_shape)

                )
            )
        )

        return regularization_loss + self.C * hinge_loss
