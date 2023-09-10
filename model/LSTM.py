import tensorflow as tf

def LSTM():
    newModel = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(384, 1)),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        # 全连接层,1 个节点
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return newModel
