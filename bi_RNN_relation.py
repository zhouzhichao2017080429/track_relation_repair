import torch
import torch.nn as nn
import numpy as np
import keras
from keras.models import Sequential,load_model,save_model
from keras.layers import Conv1D, Dense, Flatten,MaxPooling1D,Dropout
import torch
from torch_geometric.data import InMemoryDataset, Data
import h5py
from numpy import random
import math
import tensorflow as tf
import matplotlib.pyplot as plt
import os.path as osp
import scipy.io as sio
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, SimpleRNN, Activation

base_dir = "D:\实验室\项目\二院\徐老师\算法\\100条航迹\\"

# 加载.mat文件
mat_file = h5py.File(base_dir+'after_track_matrix.mat', 'r')
# 获取文件中的数据集名称
dataset_name = list(mat_file.keys())[0]
# 获取数据集
dataset = mat_file[dataset_name]
# 将数据集转换为numpy数组
after_track = dataset[()]
# 关闭文件
mat_file.close()


mat_file = h5py.File(base_dir+'before_track_matrix.mat', 'r')
dataset_name = list(mat_file.keys())[0]
dataset = mat_file[dataset_name]
before_track = dataset[()]
mat_file.close()

before_track = np.swapaxes(before_track,0,2)
after_track = np.swapaxes(after_track,0,2)

x_track_1 = np.concatenate((before_track,after_track),axis=1)

# 创建一个包含1到100的列表
lst = list(range(100))
# 随机排序列表
half = len(lst) // 2
random.shuffle(lst)
lst1 = lst[:half]
lst2 = lst[half:]

x_track_1_train = x_track_1[lst1]
x_track_1_test = x_track_1[lst2]

my_list = []
for i in range(100):
    for j in range(100):
        if i==j:
            continue
        a = np.concatenate((before_track[i,:,:],after_track[j,:,:]),axis=0)
        my_list.append(a)

random.shuffle(my_list)
x_track_0 = np.array(my_list)

x_mean_1 = np.mean(x_track_1[:,:,0],axis=1)
y_mean_1 = np.mean(x_track_1[:,:,1],axis=1)
z_mean_1 = np.mean(x_track_1[:,:,2],axis=1)

x_max_1 = np.max(x_track_1[:,:,0],axis=1)
y_max_1 = np.max(x_track_1[:,:,1],axis=1)
z_max_1 = np.max(x_track_1[:,:,2],axis=1)

x_track_1[:,:,0] = x_track_1[:,:,0] - x_mean_1[:,np.newaxis]
x_track_1[:,:,1] = x_track_1[:,:,1] - y_mean_1[:,np.newaxis]
x_track_1[:,:,2] = x_track_1[:,:,2] - z_mean_1[:,np.newaxis]

x_track_1[:,:,0] = x_track_1[:,:,0]/x_max_1[:,np.newaxis]
x_track_1[:,:,1] = x_track_1[:,:,1]/y_max_1[:,np.newaxis]
x_track_1[:,:,2] = x_track_1[:,:,2]/z_max_1[:,np.newaxis]




x_mean_0 = np.mean(x_track_0[:,:,0],axis=1)
y_mean_0 = np.mean(x_track_0[:,:,1],axis=1)
z_mean_0 = np.mean(x_track_0[:,:,2],axis=1)

x_max_0 = np.max(x_track_0[:,:,0],axis=1)
y_max_0 = np.max(x_track_0[:,:,1],axis=1)
z_max_0 = np.max(x_track_0[:,:,2],axis=1)

x_track_0[:,:,0] = x_track_0[:,:,0] - x_mean_0[:,np.newaxis]
x_track_0[:,:,1] = x_track_0[:,:,1] - y_mean_0[:,np.newaxis]
x_track_0[:,:,2] = x_track_0[:,:,2] - z_mean_0[:,np.newaxis]
#
# x_track_0[:,:,0] = x_track_0[:,:,0]/x_max_0[:,np.newaxis]
# x_track_0[:,:,1] = x_track_0[:,:,1]/y_max_0[:,np.newaxis]
# x_track_0[:,:,2] = x_track_0[:,:,2]/z_max_0[:,np.newaxis]



y_1 = np.ones((len(x_track_1),1))
y_0 = np.zeros((len(x_track_0[:500]),1))
x_track = np.concatenate((x_track_1,x_track_0[:500]),axis=0)

# y_1 = np.ones((len(x_track_1[lst1]),1))
# y_0 = np.zeros((len(x_track_0[:500]),1))
# x_track = np.concatenate((x_track_1[lst1],x_track_0[:500]),axis=0)



y_label = np.concatenate((y_1,y_0),axis=0)

def create_birnn_model(input_shape, hidden_size, output_size):
    model = Sequential()
    # model.add(LSTM(hidden_size, input_shape=(input_shape, 1)))
    # model.add(LSTM(hidden_size))
    # model.add(LSTM(hidden_size, input_shape=(input_shape, 3), return_sequences=False))
    model.add(LSTM(hidden_size, input_shape=(input_shape, 3), return_sequences=True))
    # model.add(MaxPooling1D(pool_size=4))
    # model.add(Dropout(0.3))
    model.add(Flatten())
    # model.add(Dense(10, activation='relu'))
    # model.add(Dense(10, activation='relu'))
    model.add(Dense(200))
    model.add(Dense(100))
    # model.add(Dropout(0.3))
    model.add(Dense(output_size, activation='sigmoid'))
    return model

class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size, hidden_size, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input_seq):
        # 初始化隐藏状态
        hidden = self._init_hidden()

        # 前向传播
        output, hidden = self.rnn(input_seq, hidden)

        # 双向RNN的输出是正向和反向两个方向的隐藏状态拼接而成
        # 我们将其通过全连接层进行映射得到最终输出
        output = self.fc(output)

        return output

    def _init_hidden(self):
        # 初始化隐藏状态（正向和反向两个方向的隐藏状态都需要初始化）
        return torch.zeros(2, 1, self.hidden_size)

# 模型参数设置
input_size = 30  # 输入特征维度
hidden_size = 100  # 隐藏层大小
output_size = 1  # 输出维度

# 实例化模型
model = create_birnn_model(input_size, hidden_size, output_size)

print(model.summary())

optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)


# 编译模型
model.compile(optimizer=optimizer, loss='binary_crossentropy')

model_dir = "D:\\english\\casic-2\\model\\bi-rnn-relation\\"


my_epochs = 2000

my_start = 70
my_end = 100

# model = load_model(model_dir+'model_relation.h5')


history = model.fit(x_track, y_label,validation_data=[x_track_1[lst2],np.ones((len(lst2),1))], epochs=my_epochs, batch_size=1000)
# loss_values = history.history['loss']
# # 获取训练过程中的epoch数
# epochs = range(1, len(loss_values) + 1)W


model.save(model_dir+'model_relation.h5')



predict = model.predict(x_track_1)
predict_0 = model.predict(x_track_0)

plt.figure()
plt.rcParams['font.family'] = 'Times New Roman'
ax = plt.axes(projection='3d')

nums = 10
my_start = 0
my_end = 100


# x_track_1[:,:,0] = x_track_1[:,:,0]*x_max_1[:,np.newaxis]
# x_track_1[:,:,1] = x_track_1[:,:,1]*y_max_1[:,np.newaxis]
# x_track_1[:,:,2] = x_track_1[:,:,2]*z_max_1[:,np.newaxis]

x_track_1[:,:,0] = x_track_1[:,:,0] + x_mean_1[:,np.newaxis]
x_track_1[:,:,1] = x_track_1[:,:,1] + y_mean_1[:,np.newaxis]
x_track_1[:,:,2] = x_track_1[:,:,2] + z_mean_1[:,np.newaxis]

for i in lst2:
    x = x_track_1[i,:,0]
    y = x_track_1[i,:, 1]
    z = x_track_1[i,:, 2]
    ax.plot3D(x,y,z,".",markersize=1,color="blue")

# for i in lst2:
#     x = x_track[i,:,0]
#     y = x_track[i,:, 1]
#     z = x_track[i,:, 2]
#     ax.plot3D(x,y,z,".",markersize=1,color="blue")


my_sum = 0
for i in lst2:
    a = predict[i,:]
    if a>0.9:
        my_sum = my_sum + 1
        begin_point = x_track_1[i, 14,:]
        end_point = x_track_1[i, 15,:]
        red_line = np.linspace(begin_point,end_point,10)
        ax.plot3D(red_line[:,0],red_line[:,1],red_line[:,2],".",markersize=1,color="red")

print("accuracy: ",my_sum/len(lst2))

my_sum = 0
for i in range(len(x_track_0)):
    a = predict_0[i, :]
    # print(a)
    if a < 0.1:
        my_sum = my_sum + 1
print("accuracy 2: ",my_sum/len(x_track_0))



ax.set_xlabel('x / m')
ax.set_ylabel('y / m')
ax.set_zlabel('z / m')
plt.show()