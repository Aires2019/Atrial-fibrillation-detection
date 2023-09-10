import tensorflow as tf


def SEBlock(inputs, reduction=8, if_train=True):
    x = tf.keras.layers.GlobalAveragePooling1D()(inputs)
    x = tf.keras.layers.Dense(int(x.shape[-1] // reduction), use_bias=False, activation=tf.keras.activations.relu,
                              trainable=if_train)(x)
    x = tf.keras.layers.Dense(int(inputs.shape[-1]), use_bias=False, activation=tf.keras.activations.hard_sigmoid,
                              trainable=if_train)(x)
    return tf.keras.layers.Multiply()([inputs, x])


def Seq_SE_GRU(ecg_input):
    x = tf.keras.layers.Conv1D(filters=256, kernel_size=50, strides=3, padding='same', activation=tf.nn.relu)(ecg_input)
    x = SEBlock(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2, strides=3)(x)
    x = tf.keras.layers.Conv1D(filters=128, kernel_size=7, strides=1, padding='same', activation=tf.nn.relu)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(x)
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu)(x)
    x = SEBlock(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(x)
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=10, strides=1, padding='same', activation=tf.nn.relu)(x)
    x = tf.keras.layers.Conv1D(filters=128, kernel_size=5, strides=2, padding='same', activation=tf.nn.relu)(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(x)
    x = tf.keras.layers.Conv1D(filters=256, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)(x)
    x = tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)(x)
    x = SEBlock(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(x)
    x = tf.keras.layers.GRU(units=70)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=45, activation=tf.nn.relu)(x)
    x = tf.keras.layers.Dropout(rate=0.3)(x)
    output = tf.keras.layers.Dense(units=7, activation=tf.nn.softmax)(x)

    return output


def SeqGRU():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=3, kernel_size=20, strides=1, activation=tf.nn.relu),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool1D(pool_size=2, strides=2),
        tf.keras.layers.Conv1D(filters=6, kernel_size=10, strides=1, activation=tf.nn.relu),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool1D(pool_size=2, strides=2),
        tf.keras.layers.Conv1D(filters=6, kernel_size=5, strides=1, activation=tf.nn.relu),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool1D(pool_size=2, strides=2),
        tf.keras.layers.GRU(units=24),
        tf.keras.layers.Dense(24, activation=tf.nn.relu),
        tf.keras.layers.Dense(14, activation=tf.nn.relu),
        # tf.keras.layers.Dropout(rate=0.1),
        tf.keras.layers.Dense(7, activation=tf.nn.softmax)
    ])
