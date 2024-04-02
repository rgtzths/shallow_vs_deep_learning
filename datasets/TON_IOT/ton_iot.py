import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
import tensorflow as tf

from Util import Util

class TON_IOT(Util):

    def __init__(self):
        super().__init__("TON_IOT")


    def data_processing(self):
        dataset = f"datasets/{self.name}/data/NF-ToN-IoT-v2.csv"
        output = f"datasets/{self.name}/data"

        Path(output).mkdir(parents=True, exist_ok=True)
        data = pd.read_csv(dataset)
        data.dropna()
        y = data['Label']
        x = data.drop(columns=['Label', "IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR", "L4_DST_PORT", "Attack"])
        #print(x_.shape)
#
        #_, x, _, y = train_test_split(x_, y_, test_size=0.25, stratify=y_, random_state=42)
        #del x_, y_
        
        n_samples = x.shape[0]

        scaler = QuantileTransformer(output_distribution='normal')
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)
        x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
        x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)

        print(f"\nTotal samples {n_samples}")
        print(f"Shape of the train data: {x_train.shape}")
        print(f"Shape of the test data: {x_test.shape}\n")

        # Save the data
        x_train.to_csv(f"{output}/X_train.csv", index=False)
        x_test.to_csv(f"{output}/X_test.csv", index=False)
        y_train.to_csv(f"{output}/y_train.csv", index=False)
        y_test.to_csv(f"{output}/y_test.csv", index=False)


    def create_model(self):
        model=  tf.keras.models.Sequential([
            # input layer
            tf.keras.layers.InputLayer(input_shape=(39,)),
            # hidden layers
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            # output layer
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        model.compile(
                    optimizer=tf.keras.optimizers.Adam(), 
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                    metrics=['accuracy']
                )

        return model