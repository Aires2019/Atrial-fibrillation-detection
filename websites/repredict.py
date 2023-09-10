# 用于重新加载训练结果并进行预测
from keras.models import load_model
import numpy as np


def repredict(filename):
    model = load_model("../save/CNN_LSTM_10.h5")
    file_path = 'static/' + filename
    x = np.loadtxt(file_path)
    length = x.shape[0]
    print(length)
    if length > 384:
        x = x[:384]
        print("1", x.shape)
    elif length < 384:
        x = np.pad(x, (0, 384 - length), 'constant', constant_values=(0, 0))
        print("2", x.shape)
    # 测试
    # x = np.loadtxt(filename)
    x = np.expand_dims(x, axis=0)
    data = x.reshape(384)
    # 预测
    pred = model.predict(x)
    res = round(pred[0][0], 5)
    print(res)
    if res > 0.5:
        print("检测者可能患有房颤，患病概率为{}%".format(res * 100))
        return 'abnormal', round(res * 100, 5), data
    else:
        print("检测者大概率不患有房颤，健康概率为{}%".format((1 - res) * 100))
        return 'normal', round((1 - res) * 100, 5), data


# if __name__ == "__main__":
#     a = {}
#     a = repredict('abnormal_sample1.csv')
#     print(a)
