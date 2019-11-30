# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 18:10:01 2019

@author: Gabriel Nogueras
"""

import tensorflow as tf
from tensorflow import keras


#import fashion data image set for shoes
fashion_mnist = keras.datasets.fashion_mnist


(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()



model = keras.Sequential([
        #inputs images that are 28x28 from the fashion_mnist training data set
        keras.layers.Flatten(input_shape=(28, 28)),
        #outputs a value 1-10 (number of different items of clothing)
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
        ])