#!/usr/bin/env python3
# coding: utf-8

__author__ = 'Rafael Teixeira'
__version__ = '0.1'
__email__ = 'rafaelgteixeira@ua.pt'
__status__ = 'Development'

import os
import warnings
import argparse
import pathlib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
os.environ["PYTHONWARNINGS"] = "ignore"

from sklearn.model_selection import KFold
from models import dnn_1, dnn_2, dnn_3
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import exectime.timeit as timeit

# Also affect subprocesses

@timeit.exectime(5)
def fit(cls, X, y):
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=0)
    cls.fit(X, y, epochs=50, verbose=0, callbacks=[callback])


@timeit.exectime(5)
def predict(cls, X):
    return cls.predict(X, verbose=0)


def cross_validate(X, y, models):
    k_fold = KFold(n_splits=5)
    results = [[],[],[]]
    split = 1
    for train_indices, test_indices in k_fold.split(X):
        print(f"Training split {split} of 5")
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        y_train = tf.keras.utils.to_categorical(y_train)
        for idx, model in enumerate(models):
            callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=0)
            model.fit(X_train, y_train, epochs=50, verbose=2, callbacks=[callback])
            predictions = [np.argmax(x) for x in model.predict(X_test, verbose=0)]
            mcc = matthews_corrcoef(y_test, predictions)
            results[idx].append(mcc)

        split += 1


def train_models(X_train, y_train, X_test, y_test, results_folder):
    results_file = open(results_folder/"results.md", "w")
    y_train = tf.keras.utils.to_categorical(y_train)

    models = [dnn_1(X_train.shape[1:]), dnn_2(X_train.shape[1:]), dnn_3(X_train.shape[1:])]
    print(f'| Model name | Train time | Infer time | ACC | F1  | MCC |')
    print(f'| ---------- | ---------- | ---------- | --- | --- | --- |')
    results_file.write(f'| Model name | Train time | Infer time | ACC | F1  | MCC |\n')
    results_file.write(f'| ---------- | ---------- | ---------- | --- | --- | --- |\n')
    for idx, model in enumerate(models):
        mtt, std_tt , _ = fit(model, X_train, y_train)
        mti, std_ti , y_pred = predict(model, X_test)
        y_pred = [np.argmax(y) for y in y_pred]
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        mcc = matthews_corrcoef(y_test, y_pred)
        print(f'| {idx:<10} | {round(mtt,2):>4}±{round(std_tt,2):<5} | {round(mti,2):<4}±{round(std_tt,2):<5} | {round(acc,2):<3} | {round(f1,2):<3} | {round(mcc,2):<3} |')
        results_file.write(f'| {idx:<10} | {round(mtt,2):>4}±{round(std_tt,2):<5} | {round(mti,2):<4}±{round(std_tt,2):<5} | {round(acc,2):<3} | {round(f1,2):<3} | {round(mcc,2):<3} |\n')
        model.save(results_folder/f"model_{idx}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train/test the DNNs.')
    parser.add_argument('-d', type=str, help='Dataset folder', default='dataset/processed_data/')
    parser.add_argument('-s', type=int, help='Seed used for data shuffle', default=42)
    parser.add_argument('-r', type=int, help='Results folder', default='results')
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(args.s)
    dataset = pathlib.Path(args.d)
    results = pathlib.Path(args.r)
    results.mkdir(parents=True, exist_ok=True)

    X_train = np.loadtxt(dataset/"x_train.csv", delimiter=",", dtype=int)
    y_train = np.loadtxt(dataset/"y_train.csv", delimiter=",", dtype=int)
    X_test = np.loadtxt(dataset/"x_test.csv", delimiter=",", dtype=int)
    y_test = np.loadtxt(dataset/"y_test.csv", delimiter=",", dtype=int)
    train_models(X_train, y_train, X_test, y_test, results)