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
