from typing import Union, Callable

import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk

from ..models.dnn_svm import DnnSvm


def build_model(
        n_class: int,
        model: tfk.Model = None,
        n_in: int = 2,
        n_h: int = 256,
        activation: Union[str, Callable] = 'relu',
        kernel_initializer: [str, Callable] = 'he_uniform',
        penalty=1.,
        softmax=False
):
    if softmax:
        output_function = 'softmax'
    else:
        output_function = 'linear'

    if model is None:
        model = tfk.Sequential(
            [
                tfk.layers.Dense(
                    n_h,
                    kernel_initializer=kernel_initializer,
                    input_shape=(n_in,),
                    activation=activation
                ),
                tfk.layers.Dense(
                    n_class,
                    kernel_initializer=kernel_initializer,
                    activation=output_function
                )
            ]
        )

    if softmax:
        return model
    else:
        return DnnSvm(model=model, penalty=penalty, n_class=n_class)


def set_kernel(
        model: DnnSvm,
        kernel: str = 'linear',
        degree: int = 3,
        gamma=1.,
        r=0.
):
    if kernel == 'linear':
        model.kernel = model.kernel_linear
        model.loss = model.loss_l2_linear_svm

    elif kernel == 'poly':
        model.degree = tf.Variable(
            np.array(degree).astype(np.float32), name='degree', trainable=False
        )
        model.gamma = tf.Variable(
            np.array(gamma).astype(np.float32), name='gamma', trainable=False
        )
        model.r = tf.Variable(
            np.array(r).astype(np.float32), name='r', trainable=False
        )
        model.kernel = model.kernel_poly
        model.loss = model.loss_l2_poly_svm

    elif kernel == 'rbf':
        model.gamma = tf.Variable(
            np.array(gamma).astype(np.float32), name='gamma', trainable=False
        )
        model.kernel = model.kernel_rbf
        model.loss = model.loss_l2_rbf_svm

    else:
        msg = f'Expected kernel [linear, poly, rbf], actual {kernel}'
        raise ValueError(msg)
