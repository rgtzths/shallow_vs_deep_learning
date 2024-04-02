import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import tensorflow as tf

from Util import Util

class NetSlice5G(Util):

    def __init__(self):
        super().__init__("NetSlice5G")


    def data_processing(self):
        df_train = pd.read_csv(f"datasets/{self.name}/data/train_dataset.csv")
        #df_test = pd.read_csv(f"datasets/{self.name}/data/test_dataset.csv")

        output = f"datasets/{self.name}/data"

        Path(output).mkdir(parents=True, exist_ok=True)

        x_train = df_train.values[:,:-1]
        y_train = df_train.values[:,-1]

        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, random_state=42, test_size=0.2)


        print(f"\nTotal samples {df_train.values.shape[0]}")
        print(f"Shape of the train data: {x_train.shape}")
        print(f"Shape of the test data: {x_test.shape}\n")
        
        np.savetxt(f"{output}/X_train.csv", x_train, delimiter=",", fmt="%d")
        np.savetxt(f"{output}/X_test.csv", x_test, delimiter=",", fmt="%d")
        np.savetxt(f"{output}/y_train.csv", y_train, delimiter=",", fmt="%d")
        np.savetxt(f"{output}/y_test.csv", y_test, delimiter=",", fmt="%d")

    def load_training_data(self):
        """
        Load the training data
        """
        x_train = np.loadtxt(f"datasets/{self.name}/data/X_train.csv", delimiter=",")
        y_train = np.loadtxt(f"datasets/{self.name}/data/y_train.csv", delimiter=",")
        return x_train, y_train

    def load_test_data(self):
        """
        Load the test data
        """
        x_test = np.loadtxt(f"datasets/{self.name}/data/X_test.csv", delimiter=",")
        y_test = np.loadtxt(f"datasets/{self.name}/data/y_test.csv", delimiter=",")
        return x_test, y_test
    
    def create_model(self):
        model= tf.keras.models.Sequential([
                # flatten layer
                tf.keras.layers.Flatten(input_shape=(16,)),
                # hidden layers
                tf.keras.layers.Dense(8, activation='relu'),
                tf.keras.layers.Dense(4, activation='relu'),
                tf.keras.layers.Dense(3, activation='tanh'),
                # output layer
                tf.keras.layers.Dense(3, activation="softmax")
            ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(), 
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

        return model