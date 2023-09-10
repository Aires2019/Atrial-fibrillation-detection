import tensorflow as tf


def CNN_LSTM():
    return tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(384, 1)),
        tf.keras.layers.Conv1D(filters=128, kernel_size=20, strides=3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool1D(pool_size=2, strides=3),
        tf.keras.layers.Conv1D(filters=32, kernel_size=7, strides=1, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool1D(pool_size=2, strides=2),
        tf.keras.layers.Conv1D(filters=32, kernel_size=10, strides=1, padding='same', activation='relu'),
        # tf.keras.layers.Conv1D(filters=128, kernel_size=5, strides=2, padding='same', activation=tf.nn.relu),
        tf.keras.layers.MaxPool1D(pool_size=2, strides=2),
        # tf.keras.layers.Conv1D(filters=512, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu),
        # tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu),
        tf.keras.layers.LSTM(10),
        tf.keras.layers.Flatten(),
        # tf.keras.layers.Dense(units=512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(rate=0.1),
        tf.keras.layers.Dense(units=20, activation='relu'),
        tf.keras.layers.Dense(units=10, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])


def CNN_LSTM1():
    return tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(384, 1)),
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool1D(pool_size=2, strides=2),
        tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu'),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool1D(pool_size=2, strides=2),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])


def CNN_LSTM2():
    return tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(384, 1)),
        tf.keras.layers.Conv1D(filters=64, kernel_size=7, strides=1, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool1D(pool_size=8, strides=2),
        tf.keras.layers.Conv1D(filters=128, kernel_size=5, strides=1, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool1D(pool_size=5, strides=2),
        tf.keras.layers.Conv1D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool1D(pool_size=2, strides=2),
        tf.keras.layers.LSTM(256),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=256, activation='relu'),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])


def CNN_LSTM3():
    return tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(384, 1)),
        tf.keras.layers.Conv1D(filters=32, kernel_size=2, strides=1, padding='same', activation='relu'),
        tf.keras.layers.AvgPool1D(pool_size=2, strides=2),
        tf.keras.layers.Conv1D(filters=64, kernel_size=2, strides=1, padding='same', activation='relu'),
        tf.keras.layers.AvgPool1D(pool_size=2, strides=2),
        tf.keras.layers.Conv1D(filters=128, kernel_size=2, strides=1, padding='same', activation='relu'),
        tf.keras.layers.AvgPool1D(pool_size=2, strides=2),
        tf.keras.layers.Conv1D(filters=256, kernel_size=2, strides=1, padding='same', activation=tf.nn.relu),
        tf.keras.layers.AvgPool1D(pool_size=2, strides=2),
        tf.keras.layers.Conv1D(filters=512, kernel_size=2, strides=1, padding='same', activation=tf.nn.relu),
        tf.keras.layers.AvgPool1D(pool_size=2, strides=2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])
