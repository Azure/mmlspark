# Copyright (C) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in project root for information.

import sys

if sys.version >= '3':
    basestring = str

import pyspark
from pyspark import SparkContext
from pyspark import sql
from pyspark.sql import DataFrame
import warnings
import numpy as np
import itertools

def confusionMatrix(df, y_col, y_hat_col, labels):
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt

    if isinstance(df, pyspark.sql.dataframe.DataFrame):
        df = df.select([y_col, y_hat_col]).toPandas()

    y, y_hat = df[y_col], df[y_hat_col]
    accuracy = np.mean([1. if pred==true else 0. for (pred,true) in zip(y_hat,y)])
    cm = confusion_matrix(y, y_hat)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.text(-.3, -.55,"$Accuracy$ $=$ ${}\%$".format(round(accuracy*100,1)),fontsize=18)
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=0)
    plt.yticks(tick_marks, labels, rotation=90)
    plt.imshow(cmn, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)

    thresh = .1
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 fontsize=18,
                 color="white" if cmn[i, j] > thresh else "black")

    plt.colorbar()
    plt.xlabel("Predicted Label", fontsize=18)
    plt.ylabel("True Label", fontsize=18)

def roc(df, y_col, y_hat_col, thresh=.5):
    from sklearn.metrics import roc_curve
    import matplotlib.pyplot as plt

    if isinstance(df, pyspark.sql.dataframe.DataFrame):
        df = df.select([y_col, y_hat_col]).toPandas()

    def f2i(X):
        return [int(x>thresh) for x in X]

    y, y_hat = f2i(df[y_col]), df[y_hat_col]
    fpr, tpr, thresholds = roc_curve(y, y_hat)
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate",  fontsize=20)
    plt.ylabel("True Positive Rate",  fontsize=20)
