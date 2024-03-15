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
        df_test = pd.read_csv(f"datasets/{self.name}/data/test_dataset.csv")

        output = f"datasets/{self.name}/data"

        Path(output).mkdir(parents=True, exist_ok=True)

        x_train = df_train.values[:,:-1]
        y_train = df_train.values[:,-1]

        x_test = df_test.values[:,:-1]
        y_test = df_test.values[:,-1]


        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, random_state=42, test_size=0.2)

        print(f"\nTotal samples {df_train.values.shape[0]+df_test.values.shape[0]}")
        print(f"Shape of the train data: {x_train.shape}")
        print(f"Shape of the validation data: {x_val.shape}")
        print(f"Shape of the test data: {x_test.shape}\n")
        
        np.savetxt(f"{output}/x_train.csv", x_train, delimiter=",", fmt="%d")
        np.savetxt(f"{output}/x_val.csv", x_val, delimiter=",", fmt="%d")
        np.savetxt(f"{output}/x_test.csv", x_test, delimiter=",", fmt="%d")
        np.savetxt(f"{output}/y_train.csv", y_train, delimiter=",", fmt="%d")
        np.savetxt(f"{output}/y_val.csv", y_val, delimiter=",", fmt="%d")
        np.savetxt(f"{output}/y_test.csv", y_test, delimiter=",", fmt="%d")


#    def create_model(self):
#        return tf.keras.models.Sequential([
#            # input layer
#            tf.keras.layers.InputLayer(input_shape=(39,)),
#            # hidden layers
#            tf.keras.layers.Dense(32, activation='relu'),
#            tf.keras.layers.Dropout(0.2),
#            tf.keras.layers.Dense(16, activation='relu'),
#            tf.keras.layers.Dropout(0.2),
#            # output layer
#            tf.keras.layers.Dense(2, activation='softmax')
#        ])