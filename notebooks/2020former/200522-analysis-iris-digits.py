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

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.externals import joblib

# +
out = Path('./0522_analysis')
if not out.exists():
    out.mkdir(parents=True)

in_1 = Path('./data/0518_iris_svm')
in_2 = Path('./data/0518_iris_softmax')
in_3 = Path('./data/0520_digits_svm')
in_4 = Path('./data/0520_digits_softmax')
in_5 = Path('./data/0521_digits_cnn_svm')
in_6 = Path('./data/0521_digits_cnn_softmax')
# -

sns.set()


def print_performance(path):
    df_train = pd.read_csv(path.joinpath('train-performance.csv'))
    df_test = pd.read_csv(path.joinpath('test-performance.csv'))
    print('[Train]\n')
    print(df_train)
    print('\n[Test]\n')
    print(df_test)



def concat_graph(path_1, path_2):
    df_1 = pd.read_csv(path_1.joinpath('train.log.csv')).drop('epoch', axis=1)
    df_2 = pd.read_csv(path_2.joinpath('train.log.csv')).drop('epoch', axis=1)
    for df_k, name in zip([df_1, df_2], ['svm', 'softmax']):
        df_k.rename(
            columns={
                'acc': name + '-train-accuracy',
                'loss': name + '-train-loss',
                'val_acc': name + '-validation-accuracy',
                'val_loss': name + '-validation-loss'
            },
            inplace=True
        )
    df_acc = pd.concat(
        (
            df_1.loc[:, df_1.columns.str.endswith('accuracy')],
            df_2.loc[:, df_2.columns.str.endswith('accuracy')]
        )
    )
    df_loss = pd.concat(
        (
            df_1.loc[:, df_1.columns.str.endswith('loss')],
            df_2.loc[:, df_2.columns.str.endswith('loss')]
        )
    )
    
    return (df_acc, df_loss)


# ## Performance

# ### `Iris`

# ### -svm

print_performance(in_1)

# ### -softmax

print_performance(in_2)

# ### `Digits`

# ### -svm

print_performance(in_3)

# ### -softmax

print_performance(in_4)

# ### -cnn_svm

print_performance(in_5)

# ### -cnn_softmax

print_performance(in_6)

# ## Accuracy and Loss graph

# ### Iris

df_iris_acc, df_iris_loss = concat_graph(in_1, in_2)

sns.lineplot(data=df_iris_acc)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.savefig(out.joinpath('iris-acc.pdf'))

sns.lineplot(data=df_iris_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig(out.joinpath('iris-loss.pdf'))

# ### Digits

df_digits_acc, df_digits_loss = concat_graph(in_3, in_4)

sns.lineplot(data=df_digits_acc)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.savefig(out.joinpath('digits-acc.pdf'))

sns.lineplot(data=df_digits_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig(out.joinpath('digits-loss.pdf'))

# ### Digits cnn

df_digits_cnn_acc, df_digits_cnn_loss = concat_graph(in_5, in_6)

sns.lineplot(data=df_digits_cnn_acc)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.savefig(out.joinpath('digits-cnn-acc.pdf'))

sns.lineplot(data=df_digits_cnn_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig(out.joinpath('digits-cnn-loss.pdf'))
