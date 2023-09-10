import numpy as np
import wfdb
record=wfdb.rdrecord('D:/学习/毕业设计//data//abnormal/04048',channel_names=['ECG1'])
data = record.p_signal.flatten()
annotation = wfdb.rdann('D:/学习/毕业设计//data//abnormal/04048', 'atr',
                        sampfrom=0,sampto=None,return_label_elements=['symbol'],
                        summarize_labels=False)
Rlocation = annotation.sample
Rclass = annotation.symbol
print(Rlocation)
print(Rclass)
print(annotation.aux_note)


