# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
from pathlib import Path
from typing import Union, Callable

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow.keras as tfk
from sklearn import datasets
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from dlsvm.models import DnnSvm
from dlsvm.tools import build_model, label_2_onehot
# -

out = Path('./0521_digits_cnn_softmax')
if not out.exists():
    out.mkdir(parents=True)

sns.set()

# ### Load dataset

digits = datasets.load_digits()

digits = datasets.load_digits()
X = digits.data.astype(np.float32).reshape(-1, 8, 8, 1)
y = digits.target.astype(np.float32)

# +
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_, X_train_val, y_train_, y_train_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
# -

y_train_ = label_2_onehot(y_train_)
y_train_val = label_2_onehot(y_train_val)

# ### Build model

n_class=len(digits.target_names)
n_h=64
activation='relu'
kernel_initializer='he_uniform'
output_function = 'softmax'

cnn = tfk.Sequential(
    [
        tfk.layers.Conv2D(
            input_shape=(8, 8, 1),
            filters=16,
            kernel_size=(3, 3),
            padding='same',
        ),
        tfk.layers.Flatten(),
        tfk.layers.Dense(
            n_h,
            kernel_initializer=kernel_initializer,
            activation=activation
        ),
        tfk.layers.Dense(
            n_class,
            kernel_initializer=kernel_initializer,
            activation=output_function
        )
    ]
)

cnn.summary()

dlsvm = build_model(
    n_class=n_class,
    model=cnn,
    penalty=1.,
    softmax=True
)

batch_size = 64
epochs = 50
learning_rate = 1e-3
patience = 5
loss = 'categorical_crossentropy'

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
dlsvm.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

es_cb = tfk.callbacks.EarlyStopping(
    monitor='val_loss', patience=patience, mode='auto'
)
tb_cb = tfk.callbacks.TensorBoard(log_dir=str(out))
cl_cb = tfk.callbacks.CSVLogger(
    str(out.joinpath('train.log.csv')), separator=',', append=False
)
cp_cb = tfk.callbacks.ModelCheckpoint(
    str(out.joinpath('model.weights.h5')), monitor='val_acc', verbose=0,
    save_best_only=True, save_weights_only=False, mode='auto', period=1
)

# ### Train model

history = dlsvm.fit(
    *(X_train_, y_train_),
    validation_data=(X_train_val, y_train_val),
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[tb_cb, cl_cb, cp_cb]
)

df_result = pd.DataFrame(history.history)

sns.lineplot(data=df_result[['acc', 'val_acc']])
plt.xlabel('epoch')
plt.ylabel('acc')
plt.savefig(out.joinpath('acc_logs.pdf'))

sns.lineplot(data=df_result[['loss', 'val_loss']])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig(out.joinpath('loss_logs.pdf'))

dlsvm.load_weights(str(out.joinpath('model.weights.h5')))

dlsvm.summary()

# ### Performance of model

df_train = pd.DataFrame(
    classification_report(y_pred=np.argmax(dlsvm.predict(X_train), axis=-1), y_true=y_train, output_dict=True)
)
print('Train perfomance of model')
print(df_train)
df_train.to_csv(out.joinpath('train-performance.csv'))

df_test = pd.DataFrame(
    classification_report(y_pred=np.argmax(dlsvm.predict(X_test), axis=-1), y_true=y_test, output_dict=True)
)
print('Test perfomance of model')
print(df_test)
df_test.to_csv(out.joinpath('test-performance.csv'))

np.argmax(dlsvm.predict(X_train), axis=-1)

np.argmax(dlsvm.predict(X_test), axis=-1)
