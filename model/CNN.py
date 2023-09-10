import tensorflow as tf


def test():
    newModel = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(384, 1)),
        # 打平层,方便全连接层处理'
        tf.keras.layers.Flatten(),
        # 全连接层,1 个节点
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return newModel


def CNN():
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
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=256, activation='relu'),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])


def CNN_VGG16():
    newModel = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(384, 1)),
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same'),

        tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same'),

        tf.keras.layers.Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same'),

        tf.keras.layers.Conv1D(filters=512, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv1D(filters=512, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv1D(filters=512, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same'),

        tf.keras.layers.Conv1D(filters=512, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv1D(filters=512, kernel_size=1, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv1D(filters=512, kernel_size=1, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same'),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return newModel


def CNN1d():
    newModel = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(384, 1)),
        # 第一个卷积层, 4 个 21x1 卷积核
        tf.keras.layers.Conv1D(filters=4, kernel_size=21, strides=1, padding='SAME', activation='tanh'),
        # 第一个池化层, 最大池化,4 个 3x1 卷积核, 步长为 2
        tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='SAME'),
        # 第二个卷积层, 16 个 23x1 卷积核
        tf.keras.layers.Conv1D(filters=16, kernel_size=23, strides=1, padding='SAME', activation='relu'),
        # 第二个池化层, 最大池化,4 个 3x1 卷积核, 步长为 2
        tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='SAME'),
        # 第三个卷积层, 32 个 25x1 卷积核
        tf.keras.layers.Conv1D(filters=32, kernel_size=25, strides=1, padding='SAME', activation='tanh'),
        # 第三个池化层, 平均池化,4 个 3x1 卷积核, 步长为 2
        tf.keras.layers.AvgPool1D(pool_size=3, strides=2, padding='SAME'),
        # 第四个卷积层, 64 个 27x1 卷积核
        tf.keras.layers.Conv1D(filters=64, kernel_size=27, strides=1, padding='SAME', activation='relu'),
        # 打平层,方便全连接层处理'
        tf.keras.layers.Flatten(),
        # 全连接层,128 个节点 转换成128个节点
        tf.keras.layers.Dense(128, activation='relu'),
        # Dropout层,dropout = 0.2
        tf.keras.layers.Dropout(rate=0.2),
        # 全连接层,5 个节点
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return newModel


def SeqCNN():
    return tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(384, 1)),
        # 第一个卷积层 128个 50x1 的卷积核 步长为3 进行边缘填充
        tf.keras.layers.Conv1D(filters=128, kernel_size=50, strides=3, padding='same', activation='relu'),
        # 标准化
        tf.keras.layers.BatchNormalization(),
        # 第一个池化层, 最大池化,128个 2x1 卷积核, 步长为 3
        tf.keras.layers.MaxPool1D(pool_size=2, strides=3),
        # 第二个卷积层 32个 7x1 的卷积核 步长为1
        tf.keras.layers.Conv1D(filters=32, kernel_size=7, strides=1, padding='same', activation='relu'),
        # 标准化
        tf.keras.layers.BatchNormalization(),
        # 第二个池化层 最大池化 32个 2x1 卷积核 步长为3
        tf.keras.layers.MaxPool1D(pool_size=2, strides=2),
        # 第三个卷积层 32个 10x1 的卷积核 步长为1
        tf.keras.layers.Conv1D(filters=32, kernel_size=10, strides=1, padding='same', activation='relu'),
        # 第四个卷积层 128个 5x1 的卷积核 步长为2
        tf.keras.layers.Conv1D(filters=128, kernel_size=5, strides=2, padding='same', activation='relu'),
        # 第三个池化层 最大池化 128个 2x1 卷积核 步长为2
        tf.keras.layers.MaxPool1D(pool_size=2, strides=2),
        # 第五个卷积层 512个 5x1 的卷积核 步长为1
        tf.keras.layers.Conv1D(filters=512, kernel_size=5, strides=1, padding='same', activation='relu'),
        # 第六个卷积层 128个 3x1 的卷积核 步长为1
        tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu'),
        # 打平层 方便全连接层处理
        tf.keras.layers.Flatten(),
        # 全连接层 512个节点
        tf.keras.layers.Dense(units=512, activation='relu'),
        # Dropout层,dropout = 0.2
        tf.keras.layers.Dropout(rate=0.1),
        # 全连接层 1个节点
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])
