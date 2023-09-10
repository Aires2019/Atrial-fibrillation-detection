import random
import wfdb
import pywt
import os
import numpy as np
import scipy
from scipy.signal import butter, lfilter
from scipy import signal
from scipy import stats
from imblearn.under_sampling import RandomUnderSampler

'''
此处的数据处理方式：
1、将所有数据一起提取出来，然后进行训练集和测试集的随机划分
2、进行低通滤波和小波变换
'''

# 数据位置
root_folder_train = '../data/Train/'
root_folder_test = '../data/Test/'
category = ['normal', 'abnormal']
#destination = 'D:/学习/毕业设计/data/'
destination = 'data/'

# 测试集在数据集中所占的比例
RATIO = 0.2
# 数据库的原始频率
fs_afdb = 250
fs_nsrdb =128
left_len = 1
right_len = 2 # 数据左取1s 右取2s

# 低通滤波去除高频噪音
def filtering(data, record):
    # 设置滤波器参数
    nyquist_freq = 0.5 * record.fs
    cutoff_freq = 35  # 设置截止频率为35Hz
    filter_order = 4  # 设置滤波器阶数为4

    # 计算滤波器系数
    b, a = butter(filter_order, cutoff_freq / nyquist_freq, btype='low')

    # 对信号进行滤波
    data_filtered = lfilter(b, a, data)

    # 打印原始信号和滤波后的信号的长度
    # print('Original signal length:', len(data), 'samples')
    # print('Filtered signal length:', len(data_filtered), 'samples')
    return data_filtered


# 小波阈值去噪函数
def wavelet_denoising(data):
    # 小波分解
    coeffs = pywt.wavedec(data, 'db4', level=4)
    # 小波重构
    denoised_data = pywt.waverec(coeffs, 'db4')
    return denoised_data


# 正常数据和房颤数据处理不一样
def getDataSet_abnormal(number, X_data, Y_data, flag):
    # 读取心电数据记录
    print(flag, "正在读取 ,", number, " 号心电数据...")
    record = wfdb.rdrecord(destination + category[flag] + '/' + number, channel_names=['ECG1'])
    data = record.p_signal.flatten()
    # data1 = filtering(data, record)  # 低通滤波
    # data2 = wavelet_denoising(data=data1)  # 小波变换
    res_data = stats.zscore(data)

    # 获取心电数据记录中R波的位置和对应的标签
    annotation = wfdb.rdann(destination + category[flag] + '/' + number, 'atr')
    Rlocation = annotation.sample

    # 获取标记符号列表
    symbol_list = annotation.aux_note

    # 查找所有包含N标记的注释区间
    N_ranges = []
    N_start = None
    for i, symbol in enumerate(symbol_list):
        if symbol == '(N' and N_start is None:
            N_start = Rlocation[i]
        elif symbol != '(N' and N_start is not None:
            N_end = Rlocation[i]
            N_ranges.append((N_start, N_end))
            N_start = None

    # 如果最后一个注释是N，需要手动添加其结束位置
    if N_start is not None:
        N_end = len(res_data)
        N_ranges.append((N_start, N_end))

    # 输出找到的N区间的数量
    # print('Found %d N segments.' % len(N_ranges))

    # 从正常位置中找到R峰
    for start, end in N_ranges:
        x_ann = wfdb.rdann(destination + category[flag] + '/' + number, 'qrs', sampfrom=start, sampto=end)
        Rlocation_N = x_ann.sample
        i = 5
        j = len(Rlocation_N) - 10
        while i < j:
            # 提取3s的数据
            tmp_data = res_data[Rlocation_N[i]-left_len*fs_afdb:Rlocation_N[i]+right_len*fs_afdb]
            # 需要对房颤数据集进行下采样
            re_signal = scipy.signal.resample(tmp_data, 384)  # 采样
            X_data.append(re_signal)
            Y_data.append(flag)
            i += 1
    return


# 读取心电数据和对应标签,并对数据进行小波去噪
# 传入对应的文件名，以及标注
def getDataSet_normal(number, X_data, Y_data, flag):
    # 读取心电数据记录
    print(flag, "正在读取 ,", number, " 号心电数据...")
    record = wfdb.rdrecord(destination + category[flag] + '/' + number, channel_names=['ECG1'])
    data = record.p_signal.flatten()
    # data1 = filtering(data, record)  # 低通滤波
    # data2 = wavelet_denoising(data=data1)  # 小波变换
    res_data = stats.zscore(data)

    # 获取心电数据记录中R波的位置和对应的标签
    annotation = wfdb.rdann(destination + category[flag] + '/' + number, 'atr')
    Rlocation = annotation.sample

    # 去掉前后的不稳定数据
    i = 10
    j = len(annotation.symbol) - 5

    # 正常心跳截取R峰左右，按照经验左边取100点，右边取200点
    while i < j:
        try:
            x_train = res_data[Rlocation[i] - left_len*fs_nsrdb:Rlocation[i] + right_len*fs_nsrdb]
            X_data.append(x_train)
            Y_data.append(flag)
            i += 1
        except ValueError:
            i += 1
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
def processingData_simple():
    normal_set = []
    abnormal_set = []
    loadData(destination, normal_set, abnormal_set)
    dataSet = []
    labelSet = []  # 正常心电图标签为0 心房颤动心电图标签为1
    for n in normal_set:
        getDataSet_normal(n, dataSet, labelSet, 0)
    for m in abnormal_set:
        getDataSet_abnormal(m, dataSet, labelSet, 1)

    # 转numpy数组,打乱顺序
    dataSet = np.array(dataSet).reshape(-1, 384)
    labelSet = np.array(labelSet).reshape(-1, 1)
    train_ds = np.hstack((dataSet, labelSet))
    # random.seed(42)
    np.random.shuffle(train_ds)

    # 数据集及其标签集
    X = train_ds[:, :384]
    Y = train_ds[:, 384]

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
    # 对训练集进行随机欠采样
    rus = RandomUnderSampler(random_state=42)
    X_resampled, Y_resampled = rus.fit_resample(X_train, Y_train)
    X_resampled = X_resampled.reshape(-1, 384, 1)
    X_test = X_test.reshape(-1,384,1)
    return X_resampled, Y_resampled, X_test, Y_test


# X_train, Y_train, X_test, Y_test = processingData()
# print(X_train.shape)
# print(Y_train.shape)
# print(X_test.shape)
# print(Y_test.shape)
# np.save('X_test.npy', X_test)
# np.save('Y_test.npy', Y_test)
