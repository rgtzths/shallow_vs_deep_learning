#!/usr/bin/env python3
# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mario.antunes@av.it.pt'
__status__ = 'Development'

import os
import numpy as np
import argparse
import pathlib
import exectime.timeit as timeit

import joblib
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import tensorflow as tf

import warnings
warnings.filterwarnings('ignore')
os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

from config import DATASETS


model_mapping ={'LOG': LogisticRegression,
                'KNN': KNeighborsClassifier,
                'SVM': SVC,
                'NB': GaussianNB,
                'DT': DecisionTreeClassifier,
                'RF': RandomForestClassifier,
                'ABC': AdaBoostClassifier,
                'GBC': GradientBoostingClassifier}


@timeit.exectime(5)
def fit(cls, X, y, is_sklearn):
    if is_sklearn:
        with joblib.parallel_backend(backend='loky', n_jobs=-1):
            cls.fit(X, y)
    else:
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=0)
        cls.fit(X, y, epochs=50, verbose=0, callbacks=[callback])


@timeit.exectime(5)
def predict(cls, X, is_sklearn):
    if is_sklearn:
        with joblib.parallel_backend(backend='loky', n_jobs=-1):
            return cls.predict(X)
    else:
        return cls.predict(X, verbose=0)


def optimize(cls_name, parameters, cv = 5):
    cls = model_mapping[cls_name]()
    grid = GridSearchCV(cls, param_grid=parameters, scoring='f1_weighted', cv=cv, n_jobs=1)
    grid.fit(X_train, y_train)
    cls = model_mapping[cls_name](**grid.best_params_)
    return cls


def train_models(X_train, y_train, X_test, y_test, model, seed, results_folder):
    results_file = open(results_folder/"results.md", "w")
    models = [('LOG', {'random_state':[seed], 'penalty': ['l1','l2']}),
              ('KNN', {'weights': ['uniform', 'distance'], 'n_neighbors': [3,5,7]}),
              ('SVM', {'random_state':[seed], 'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['linear', 'rbf']}),
              ('NB',  {'var_smoothing': np.logspace(0,-9, num=100)}),
              ('DT',  {'random_state':[seed], 'criterion':['gini','entropy'],'max_depth':[3,5,7,9], 'max_features': ['auto', 'sqrt', 'log2']}),
              ('RF',  {'random_state':[seed], 'n_estimators':[200,300,400], 'max_features':['auto', 'sqrt', 'log2'], 'max_depth':[3,5,7,9],}),
              ('ADC', {'random_state':[seed], 'n_estimators':[200,300,400]}),
              ('GBC', {'random_state':[seed], 'n_estimators':[200,300,400], 'max_features':['auto', 'sqrt', 'log2'], 'max_depth':[3,5,7,9]}),
              ('DNN', {})
              ]

    print(f'| Model name | Train time | Infer time | ACC | F1  | MCC |')
    print(f'| ---------- | ---------- | ---------- | --- | --- | --- |')
    results_file.write(f'| Model name | Train time | Infer time | ACC | F1  | MCC |\n')
    results_file.write(f'| ---------- | ---------- | ---------- | --- | --- | --- |\n')
    for cls_name, parameters in models:
        is_sklearn = cls_name in model_mapping
        if is_sklearn:
            cls = optimize(cls_name, parameters)
        else:
            cls = model
        mtt, std_tt , _ = fit(cls, X_train, y_train, is_sklearn)
        mti, std_ti , y_pred = predict(cls, X_test, is_sklearn)
        y_pred = y_pred if is_sklearn else [np.argmax(y) for y in y_pred]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        mcc = matthews_corrcoef(y_test, y_pred)
        
        print(f'| {cls_name:<10} | {round(mtt,2):>4}±{round(std_tt,2):<5} | {round(mti,2):<4}±{round(std_tt,2):<5} | {round(acc,2):<3} | {round(f1,2):<3} | {round(mcc,2):<3} |')
        
        results_file.write(f'| {cls_name:<10} | {round(mtt,2):>4}±{round(std_tt,2):<5} | {round(mti,2):<4}±{round(std_tt,2):<5} | {round(acc,2):<3} | {round(f1,2):<3} | {round(mcc,2):<3} |\n')
        if is_sklearn:
            joblib.dump(cls, results_folder/ f'{cls_name}.joblib')
        else:
            model.save(results_folder/f"dnn_model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train/test the DNNs.')
    parser.add_argument("-d", help=f"Dataset name {list(DATASETS.keys())}", default="IOT_DNL")
    parser.add_argument('-s', type=int, help='Seed used for data shuffle', default=42)
    parser.add_argument('-r', type=int, help='Results folder', default='results')
    args = parser.parse_args()

    if args.d not in DATASETS.keys():
        raise ValueError(f"Dataset name must be one of {list(DATASETS.keys())}")
    
    tf.keras.utils.set_random_seed(args.s)
    dataset = DATASETS[args.d]()

    results = pathlib.Path(args.r)
    results = results / dataset.name
    results.mkdir(parents=True, exist_ok=True)

    X_train, y_train = dataset.load_training_data()
    X_test, y_test = dataset.load_test_data()

    train_models(X_train, y_train, X_test, y_test, dataset.create_model(), args.s, results)