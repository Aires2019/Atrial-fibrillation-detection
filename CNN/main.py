import seaborn
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from data_processing import processingData
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from tensorflow.python.keras import Model, Input
# from tensorflow.python.keras.layers import LSTM, Bidirectional, Dropout, Dense, Attention, multiply
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.python.keras.layers.core import *
from tensorflow.python.keras.utils.vis_utils import plot_model

# 测试集在数据集中所占的比例
RATIO = 0.2


# 构建CNN模型
def buildModel():
    newModel = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(300, 1)),
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
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    return newModel


def plotHeatMap(Y_test, Y_pred):
    con_mat = confusion_matrix(Y_test, Y_pred)
    # 绘图
    plt.figure(figsize=(4, 5))
    seaborn.heatmap(con_mat, annot=True, fmt='.20g', cmap='Blues')
    plt.ylim(0, 5)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()


def main():
    # X_train,Y_train为所有的数据集和标签集
    # X_test,Y_test为拆分的测试集和标签集
    X_train, Y_train, X_test, Y_test = processingData()
    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)
    model = buildModel()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy']
                  # metrics: 列表，包含评估模型在训练和测试时的性能的指标，典型用法是metrics=[‘accuracy’]。
                  )
    # model.compile(optimizer='adam',
    #               loss='binary_crossentropy', metrics=['accuracy']
    #               # metrics: 列表，包含评估模型在训练和测试时的性能的指标，典型用法是metrics=[‘accuracy’]。
    #               )
    model.summary()

    # 训练与验证
    model.fit(X_train, Y_train, epochs=5, batch_size=64, validation_split=RATIO)  # validation_split 训练集所占比例
    # 预测
    Y_pred = model.predict(X_test)
    print('res\n', Y_pred)

    Y_true = []
    for element in Y_pred:
        if element[0] > 0.5:
            Y_true.append(1)
        else:
            Y_true.append(0)

    loss, accuracy = model.evaluate(X_test, Y_test)
    print('test loss', loss)
    print('accuracy', accuracy)

    print('Precision(精确率，预测为正常中真正正常): %.3f' % precision_score(Y_test, Y_true, average='micro'))
    print('F1 Score: %.3f' % f1_score(Y_test, Y_true, average='micro'))
    print('Recall(所有正常被预测的比例): %.3f' % recall_score(Y_test, Y_true, average='micro'))
    print('Accuracy: %.3f' % accuracy_score(Y_test, Y_true))

    # plotHeatMap(Y_test,Y_pred)


if __name__ == '__main__':
    main()
