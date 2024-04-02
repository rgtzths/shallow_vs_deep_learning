from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

from Util import Util

class IOT_DNL(Util):

    def __init__(self):
        super().__init__("IOT_DNL")



    def data_processing(self):
        dataset = f"datasets/{self.name}/data/Preprocessed_data.csv"
        output = f"datasets/{self.name}/data"
        Path(output).mkdir(parents=True, exist_ok=True)

        data = pd.read_csv(dataset)
        data.dropna()
        X = data.drop('normality', axis=1)
        X = X.drop('frame.number', axis=1)
        X = X.drop('frame.time', axis=1)
        y = data['normality']
        n_samples=X.shape[0]

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
        scaler = StandardScaler()
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
        model =  tf.keras.models.Sequential([
            # flatten layer
            tf.keras.layers.Flatten(input_shape=(11,)),
            # hidden layers
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(64, activation='relu'),
            # output layer
            tf.keras.layers.Dense(6, activation='softmax')
        ])

        model.compile(
                    optimizer=tf.keras.optimizers.Adam(), 
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                    metrics=['accuracy']
                )

        return model