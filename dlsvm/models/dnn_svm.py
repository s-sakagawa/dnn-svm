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
        self.loss = None
        self.kernel = None
        self.degree = None
        self.gamma = None
        self.r = None

        self.ex_model = tfk.Sequential(
            tfk.layers.Dense(
                n_class,
                kernel_initializer='he_uniform',  # 'identity'
                input_shape=(n_class,),
                activation='linear',
                use_bias=True
            )
        )
        self.k_sv = tf.Variable(
            np.where(np.eye(n_class) == 1, 1, -1).astype(np.float32), name='k_sv', trainable=False
        )

    @tf.function
    def call(self, inputs, **kwargs):
        h = self.model(inputs)
        k = self.kernel(h)
        outputs = self.ex_model(k)

        return outputs

    def kernel_linear(self, inputs):
        x = tf.expand_dims(self.k_sv, axis=0)
        y = tf.expand_dims(inputs, axis=1)

        return tf.squeeze(tf.matmul(x, y, transpose_b=True), axis=-1)

    def kernel_poly(self, inputs):
        x = tf.expand_dims(self.k_sv, axis=0)
        y = tf.expand_dims(inputs, axis=1)

        return (self.gamma * tf.squeeze(tf.matmul(x, y, transpose_b=True), axis=-1) + self.r) ** self.degree

    def kernel_rbf(self, inputs):
        x = tf.expand_dims(self.k_sv, axis=0)
        y = tf.expand_dims(inputs, axis=1)

        return tf.math.exp(-self.gamma * (tf.linalg.norm(tensor=x - y, axis=-1) ** 2))

    # Default loss function
    def loss_l2_svm(self, y_true, y_pred):
        matrix_shape = [tf.shape(y_true)[0], self.n_class]
        weight = self.ex_model.layers[-1].weights[0]

        regularization_loss = 0.5 * tf.reduce_mean(tf.square(weight))
        hinge_loss = tf.reduce_mean(
            tf.square(
                tf.maximum(
                    tf.ones(matrix_shape) - y_pred * y_true, tf.zeros(matrix_shape)

                )
            )
        )

        return regularization_loss + self.C * hinge_loss

    def loss_l2_linear_svm(self, y_true, y_pred):
        matrix_shape = [tf.shape(y_true)[0], self.n_class]
        weight = self.model.layers[-1].weights[0]
        kernel = y_pred

        regularization_loss = 0.5 * tf.reduce_mean(tf.square(weight))
        hinge_loss = tf.reduce_mean(
            tf.square(
                tf.maximum(
                    tf.ones(matrix_shape) - kernel * y_true, tf.zeros(matrix_shape)
                )
            )
        )

        return regularization_loss + self.C * hinge_loss

    def loss_l2_poly_svm(self, y_true, y_pred):
        matrix_shape = [tf.shape(y_true)[0], self.n_class]
        weight = self.model.layers[-1].weights[0]
        kernel = tf.math.pow(
            self.gamma * y_pred + self.r * tf.ones(matrix_shape),
            self.degree * tf.ones(matrix_shape)
        )

        regularization_loss = 0.5 * tf.reduce_mean(tf.square(weight))
        hinge_loss = tf.reduce_mean(
            tf.square(
                tf.maximum(
                    tf.ones(matrix_shape) - kernel * y_true, tf.zeros(matrix_shape)
                )
            )
        )

        return regularization_loss + self.C * hinge_loss

    def loss_l2_rbf_svm(self, y_true, y_pred):
        matrix_shape = [tf.shape(y_true)[0], self.n_class]
        weight = self.model.layers[-1].weights[0]
        kernel = tf.math.exp(
            -self.gamma * tf.math.pow(
                y_pred, 2 * tf.ones(matrix_shape)
            )
        )

        regularization_loss = 0.5 * tf.reduce_mean(tf.square(weight))
        hinge_loss = tf.reduce_mean(
            tf.square(
                tf.maximum(
                    tf.ones(matrix_shape) - kernel * y_true, tf.zeros(matrix_shape)
                )
            )
        )

        return regularization_loss + self.C * hinge_loss
