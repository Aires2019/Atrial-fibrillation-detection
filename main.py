from dataProcessing.process import processingData
from dataProcessing.process_new import processingData_new
from dataProcessing.process_simple import processingData_simple
from tensorflow.keras.optimizers import SGD,Adam
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import seaborn
import numpy as np
from model.CNN import CNN
from model.CNN import CNN1d
from model.CNN import SeqCNN
from model.CNN import CNN_VGG16
from model.CNN_LSTM import CNN_LSTM
from model.CNN_LSTM import CNN_LSTM1
from model.CNN_LSTM import CNN_LSTM2
from model.LSTM import LSTM


# 画图
def plotLoss(history, epoch):
    N = np.arange(0, epoch)
    plt.figure(0)
    plt.plot(N, history.history["loss"], label="train_loss")
    plt.plot(N, history.history["val_loss"], label="val_loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("output/loss_CNN_LSTM.png")


def plotAccuracy(history, epoch):
    N = np.arange(0, epoch)
    plt.figure(1)
    plt.plot(N, history.history["accuracy"], label="train_acc")
    plt.plot(N, history.history["val_accuracy"], label="val_acc")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("output/accuracy_CNN_LSTM.png")


def plotHeatMap(Y_test, Y_pred):
    con_mat = confusion_matrix(Y_test, Y_pred)
    # 绘图
    plt.figure(2, figsize=(4, 5))
    seaborn.heatmap(con_mat, annot=True, fmt='.20g', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.savefig("output/confusion_matrix_CNN_LSTM.png")


def plot_roc_curve(Y_test, Y_pred):
    fpr, tpr, thresholds = roc_curve(Y_test, Y_pred)
    plt.figure(3)
    plt.plot(fpr, tpr, linewidth=2, label=None)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.savefig("output/ROC_CNN_LSTM.png")


def main():
    # X_train,Y_train为所有的数据集和标签集
    # X_test,Y_test为拆分的测试集和标签集
    X_train, Y_train, X_test, Y_test = processingData_new()
    # X_train, Y_train, X_test, Y_test = processingData()
    # print("===============粗处理数据:=======================")
    # X_train = np.load('res_data/X_train_simple.npy')
    # Y_train = np.load('res_data/Y_train_simple.npy')
    # X_test = np.load('res_data/X_test_simple.npy')
    # Y_test = np.load('res_data/Y_test_simple.npy')

    print("===============划分病人，细处理数据:=======================")
    # X_train = np.load('res_data/X_train.npy')
    # Y_train = np.load('res_data/Y_train.npy')
    # X_test = np.load('res_data/X_test.npy')
    # Y_test = np.load('res_data/Y_test.npy')

    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)

    epochs = 10
    batch_size = 64

    # 使用CNN-LSTM模型
    print("=========================开始训练CNN模型...=============================")
    model = CNN()
    optimizer = SGD(learning_rate=0.0001, momentum=0.5)
    # optimizer = Adam(lr=0.001)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy', metrics=['accuracy']
                  # metrics: 列表，包含评估模型在训练和测试时的性能的指标，典型用法是metrics=[‘accuracy’]。
                  )
    model.summary()

    # 训练与验证
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_test, Y_test))  # validation_split 验证集所占比例
    # history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,
    #                     validation_split=0.2)  # validation_split 验证集所占比例

    # 预测
    print("=========================正在评估模型:===================================")
    Y_pred = model.predict(X_test)
    print('res\n', Y_pred)

    Y_true = Y_test.astype(int)
    Y_pre = []
    for element in Y_pred:
        if element[0] > 0.5:
            Y_pre.append(1)
        else:
            Y_pre.append(0)

    loss, accuracy = model.evaluate(X_test, Y_test)
    print('test loss', loss)
    print('accuracy', accuracy)

    print('Precision(精确率，预测为正常中真正正常): %.4f' % precision_score(Y_true, Y_pre))
    print('F1 Score: %.4f' % f1_score(Y_true, Y_pre))
    print('Recall(所有正常被预测的比例): %.4f' % recall_score(Y_true, Y_pre))
    print('Accuracy: %.4f' % accuracy_score(Y_true, Y_pre))

    # print("=======================保存结果===========")
    # np.savetxt("output/y_test.txt", Y_test)
    # np.savetxt("output/y_true.txt", Y_pred)

    # 绘图
    plotLoss(history, epochs)
    plotAccuracy(history, epochs)

    # ROC
    plot_roc_curve(Y_test, Y_pre)

    # 混淆矩阵
    plotHeatMap(Y_test, Y_pre)

    # 保存模型
    # print("正在保存模型...")
    # model.save('save/CNN_LSTM_10.h5')

    print("Finished!!")


if __name__ == '__main__':
    main()
