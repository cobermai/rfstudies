"""
combines all steps TRANSFORMATION, DATA HANDLING, MODELLING
"""
from pathlib import Path
from collections import namedtuple
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense, Dropout, Activation
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from src.transformation import transform
from src.handler import XBox2ContextDataCreator
from src import union
from src.xbox2_speciffic.select_features.simple_select import select_data


def transformation(work_dir: Path):
    """TRANSFORMATION"""
    src_dir = Path("~/project_data/CLIC_DATA_Xbox2_T24PSI_2/").expanduser()
    transform(tdms_dir=src_dir,
              hdf_dir=work_dir)

    gathered_trend_data = work_dir / "TrendDataExtLinks.hdf"
    combined_trend_data_path = work_dir / "combined.hdf"

    union.merge(source_file_path=gathered_trend_data,
                dest_file_path=combined_trend_data_path)
    union.convert_iso8601_to_datetime(file_path=combined_trend_data_path)
    union.sort_by(file_path=combined_trend_data_path, sort_by_name="Timestamp")


def data_handling(work_dir: Path):
    """DATA HANDLING"""
    gathered_event_data_path = work_dir / "EventDataExtLinks.hdf"
    context_data_file_path = work_dir / "context.hdf"
    combined_trend_data_path = work_dir / "combined.hdf"

    creator = XBox2ContextDataCreator(ed_file_path=gathered_event_data_path,
                                      td_file_path=combined_trend_data_path,
                                      dest_file_path=context_data_file_path)
    creator.manage_features()


def class_weights_for_onehot(y):
    y_class = np.argmax(y, axis=1)
    weights = compute_class_weight('balanced', classes=np.unique(y_class), y=y_class)
    return dict(enumerate(weights))

def get_model(train):
    input_shape = train.x.shape[1:]
    output_len = train.y.shape[1]

    input_layer = Input(shape=input_shape)
    layer1 = Dense(100, activation=tf.nn.relu)(input_layer)
    layer2 = Dense(50, activation=tf.nn.relu)(layer1)
    layer3 = Dense(20, activation=tf.nn.relu)(layer2)
    output_layer = Dense(output_len, activation=tf.nn.softmax)(layer3)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                  metrics=[keras.metrics.Accuracy(), keras.metrics.CategoricalAccuracy(), keras.metrics.AUC()])
    return model

def modelling():
    """MODELLING"""
    X, y = select_data()
    X = np.nan_to_num(X)  # tensorflow does not like nan values as input. Remove them in any kind of way
    y = y.reshape(-1, 1)
    encoder = sklearn.preprocessing.OneHotEncoder(categories='auto')
    encoder.fit(y)
    y = encoder.transform(y).toarray()

    seed = 3141592
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=seed)

    data = namedtuple("data", ["x", "y"])
    train = data(X_train, y_train)
    test = data(X_test, y_test)

    model = get_model(train)

    batch_size = 64
    num_epochs = 300
    class_weights = class_weights_for_onehot(train.y)
    hist = model.fit(train.x, train.y, batch_size=batch_size, validation_split=0.2, epochs=num_epochs,
                     class_weight=class_weights, verbose=True)

    print(model.test_on_batch(x=test.x, y=test.y, return_dict=True))


if __name__ == "__main__":
    output_files = Path("~/output_files/").expanduser()
    transformation(work_dir=output_files)
    data_handling(work_dir=output_files)
    modelling()
