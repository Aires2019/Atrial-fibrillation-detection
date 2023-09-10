import wfdb
import pywt
import os
import numpy as np
import pandas as pd

# 数据位置
root_folder_train = '../data/Train/'
root_folder_test = '../data/Test/'
category = ['normal', 'abnormal']
destination = 'D:/学习/毕业设计//data/'

# 测试集在数据集中所占的比例
RATIO = 0.2


# 小波去噪预处理
def denoise(data):
    # 小波变换
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    # 阈值去噪
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    # 小波反变换,获取去噪后的信号
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata


# 读取心电数据和对应标签,并对数据进行小波去噪
# 传入对应的文件名，以及标注
def getDataSet(number, X_data, Y_data, flag):
    # 读取心电数据记录
    print(flag, "正在读取 ,", number, " 号心电数据...")
    record = wfdb.rdrecord(destination + category[flag] + '/' + number, channel_names=['ECG1'])
    data = record.p_signal.flatten()
    rdata = denoise(data=data)

    # 获取心电数据记录中R波的位置和对应的标签
    annotation = wfdb.rdann(destination + category[flag] + '/' + number, 'atr')
    Rlocation = annotation.sample

    # 去掉前后的不稳定数据
    start = 1
    i = 10
    j = len(annotation.symbol) - 5

    # 正常心跳截取R峰左右，按照经验左边取100点，右边取200点
    if flag == 0:
        while i < j:
            try:
                x_train = rdata[Rlocation[i] - 99:Rlocation[i] + 201]
                X_data.append(x_train)
                Y_data.append(flag)
                i += 1
            except ValueError:
                i += 1
    # 阵发性房颤需要找到标记为“(N”的对应点，然后进行左右选取
    elif flag == 1:
        # 获取标记为正常的下标
        normal_list = [i for i, x in enumerate(annotation.aux_note) if x == '(N']
        end=len(normal_list)-1
        while start < end:
            try:
                x_train = rdata[Rlocation[normal_list[start]] - 99:Rlocation[normal_list[start]] + 100]
                X_data.append(x_train)
                Y_data.append(flag)
                start += 1
            except ValueError:
                start += 1
    return


def loadData(root_folder, normal_data, abnormal_data):
    for cat in category:
        folder_name = []
        for filename in os.listdir(root_folder + cat):
            # 读取目录下的所有文件
            if filename[0:5] in folder_name:  # 判断文件名的前五个字符
                continue
            else:
                folder_name.append(filename[0:5])
        for name in folder_name:
            if name[0] not in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']:  # 跳过无用的文件
                continue
            elif name in ['00735', '03665']:  # 跳过不可用文件
                continue
            if cat == 'normal':
                normal_data.append(name)
            elif cat == 'abnormal':
                abnormal_data.append(name)


# 加载数据集并进行预处理
def processingData():
    normal_set = []
    abnormal_set = []
    loadData(destination, normal_set, abnormal_set)
    dataSet = []
    labelSet = []  # 正常心电图标签为0 心房颤动心电图标签为1
    for n in normal_set:
        getDataSet(n, dataSet, labelSet, 0)
    for m in abnormal_set:
        getDataSet(m, dataSet, labelSet, 1)

    # 转numpy数组,打乱顺序
    dataSet = np.array(dataSet).reshape(-1, 300)
    labelSet = np.array(labelSet).reshape(-1, 1)
    train_ds = np.hstack((dataSet, labelSet))
    np.random.shuffle(train_ds)

    # 数据集及其标签集
    X = train_ds[:, :300].reshape(-1, 300, 1)
    Y = train_ds[:, 300]

    # 测试集及其标签集
    shuffle_index = np.random.permutation(len(X))
    # 设定测试集的大小 RATIO是测试集在数据集中所占的比例
    test_length = int(RATIO * len(shuffle_index))
    # 测试集的长度
    test_index = shuffle_index[:test_length]
    # 训练集的长度
    train_index = shuffle_index[test_length:]
    X_test, Y_test = X[test_index], Y[test_index]
    X_train, Y_train = X[train_index], Y[train_index]
    return X_train, Y_train, X_test, Y_test


normal_set = []
abnormal_set = []
loadData(destination, normal_set, abnormal_set)
print("正常心电图样本",len(normal_set))
print("房颤样本",len(abnormal_set))
data1=[]
label1=[]
for n in normal_set:
    getDataSet(n, data1, label1, 0)

data2=[]
label2=[]
for n in abnormal_set:
    getDataSet(n, data2, label2, 1)

print("正常数量",len(data1))
print(data1[10])
print("------")
print(label1[10])
print("不正常数量",len(data2))
print(data2[10])
print("------")
print(label2[10])


# X_train, Y_train, X_test, Y_test = processingData()
# print(X_train.shape)
# print(Y_train.shape)
# print(X_test.shape)
# print(Y_train.shape)
# np.save('X_test.npy', X_test)
# np.save('Y_test.npy', Y_test)
