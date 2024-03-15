import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import tensorflow as tf

from Util import Util

class NetworkSlicing5G(Util):

    def __init__(self):
        super().__init__("NetworkSlicing5G")


    def data_processing(self):
        df = pd.read_csv(f"datasets/{self.name}/data/5G_network_slice_dataset.csv")

        output = f"datasets/{self.name}/data"

        enc = OneHotEncoder(handle_unknown='error')
        le = LabelEncoder()

        Path(output).mkdir(parents=True, exist_ok=True)

        df["slice"] = le.fit_transform(df["slice"])
        encoded_features = df.drop("packet_delay_budget", axis=1)
        encoded_features = encoded_features.drop("slice", axis=1)
        encoded_features = encoded_features.drop("packet_loss_rate", axis=1)
        encoded_features = encoded_features.drop("duration_hrs", axis=1)
        
        enc.fit(encoded_features)

        data = enc.transform(encoded_features).toarray()
        data = np.append(data, df["duration_hrs"].to_numpy().reshape(-1,1), axis=1)
        data = np.append(data, df["packet_loss_rate"].to_numpy().reshape(-1,1), axis=1)
        data = np.append(data, df["packet_delay_budget"].to_numpy().reshape(-1,1), axis=1)
        data = np.append(data, df["slice"].to_numpy().reshape(-1,1), axis=1)

        x = data[:,:-1]
        y = data[:,-1]

        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, random_state=42, test_size=0.2)
        
        print(f"\nTotal samples {df.values.shape[0]}")
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