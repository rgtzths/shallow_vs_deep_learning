#!/usr/bin/env python3
# coding: utf-8

__author__ = 'Rafael Teixeira'
__version__ = '0.1'
__email__ = 'rafaelgteixeira@ua.pt'
__status__ = 'Development'

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential


def dnn_1(input_shape): #https://ieeexplore.ieee.org/document/10104899
    model = Sequential()
    model.add(Dense(32,input_shape=input_shape, activation="relu"))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(4, activation="tanh"))
    model.add(Dense(3, activation="softmax"))  

    model.compile(optimizer="adam", loss="categorical_crossentropy")

    return model


def dnn_2(input_shape): #https://ieeexplore.ieee.org/document/10008652
    model = Sequential()
    model.add(Dense(32, activation="relu", input_shape=input_shape))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(3, activation="softmax"))
    model.compile(optimizer="adam", loss='categorical_crossentropy')

    return model

#Original network
def dnn_3(input_shape): #https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8993066
    model = Sequential()
    model.add(Dense(8, activation="relu", input_shape=input_shape))
    model.add(Dense(4, activation="relu"))
    model.add(Dense(3, activation="tanh"))
    model.add(Dense(3, activation="softmax"))
    model.compile(optimizer="adam", loss='categorical_crossentropy')

    return model

