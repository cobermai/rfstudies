import h5py
import numpy as np
from pathlib import Path
import pandas as pd
#from tensorflow import keras
#from tensorflow.keras import layers
from src.utils.hdf_tools import hdf_to_df, hdf_to_df_selection

context_data_file_path = Path("~/output_files/contextd.hdf").expanduser()


with h5py.File(context_data_file_path, "r") as file:
    selection = np.logical_or(file["is_bd_in_40ms"][:], file["is_bd_in_20ms"][:], file["is_log"][:])

    #querying datetime is not possible. See pandas numexpr.necompiler.getType
    event_ts = file["Timestamp"][:]
    trend_ts = file["PrevTrendData/Timestamp"][:]
    diff = event_ts - trend_ts
    threshold = diff.sort_values(ascending=True)[int(len(diff) / 40)]
    threshold = pd.to_timedelta(2,"s")
    filter_timestamp_diff = diff < threshold

    selection = np.logical_and(selection, filter_timestamp_diff)

    is_log = file["is_log"]
    selection[is_log] = np.random.choice(a=[True, False],
                                         size=(sum(is_log),),
                                         p=[0.025, 0.975])  # select 2.5% of log signals randomly

#df = hdf_to_df(context_data_file_path)
df = hdf_to_df_selection(context_data_file_path, selection=selection)

clm_for_training = df.columns.difference(pd.Index(["Timestamp", "PrevTrendData__Timestamp", "is_bd", "is_log", "is_bd_in_20ms", "is_bd_in_40ms"]))
X = df[clm_for_training].to_numpy(dtype=float)
Y = df["is_log"].to_numpy(dtype=int)








#
#model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
#              loss=keras.losses.CategoricalCrossentropy())
#
## Get the data as Numpy arrays
#(x_train, y_train), (x_test, y_test) =
#
## Build a simple model
#inputs = keras.Input(shape=(28, 28))
#x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
#x = layers.Flatten()(x)
#x = layers.Dense(128, activation="relu")(x)
#x = layers.Dense(128, activation="relu")(x)
#outputs = layers.Dense(10, activation="softmax")(x)
#model = keras.Model(inputs, outputs)
#model.summary()
#
## Compile the model
#model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
#
## Train the model for 1 epoch from Numpy data
#batch_size = 64
#print("Fit on NumPy data")
#history = model.fit(x_train, y_train, batch_size=batch_size, epochs=1)
#
## Train the model for 1 epoch using a dataset
#dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
#print("Fit on Dataset")
#history = model.fit(dataset, epochs=1)
