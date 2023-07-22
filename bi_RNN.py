import torch
import torch.nn as nn
import numpy as np
import keras
from keras.models import Sequential,load_model,save_model
from keras.layers import Conv1D, Dense, Flatten, Dropout, Reshape
import torch
from torch_geometric.data import InMemoryDataset, Data
import h5py
from numpy import random
import math
import tensorflow as tf
import matplotlib.pyplot as plt
import os.path as osp
from projection_dist import projection_dist
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

mat_file = h5py.File(base_dir+'miss_track_matrix.mat', 'r')
dataset_name = list(mat_file.keys())[0]
dataset = mat_file[dataset_name]
miss_track = dataset[()]
mat_file.close()

before_track = np.swapaxes(before_track,0,2)
after_track = np.swapaxes(after_track,0,2)
miss_track = np.swapaxes(miss_track,0,2)

x_track = np.concatenate((before_track,after_track),axis=1)
after_track_shuffle = after_track.copy()
random.shuffle(after_track_shuffle)
test_track_incorrect = np.concatenate((before_track,after_track_shuffle),axis=1)
# x_track = before_track
y_track = miss_track.copy()

x_mean = np.mean(x_track[:,:,0],axis=1)
y_mean = np.mean(x_track[:,:,1],axis=1)
z_mean = np.mean(x_track[:,:,2],axis=1)

x_mean_test = np.mean(test_track_incorrect[:,:,0],axis=1)
y_mean_test = np.mean(test_track_incorrect[:,:,1],axis=1)
z_mean_test = np.mean(test_track_incorrect[:,:,2],axis=1)

x_track[:,:,0] = x_track[:,:,0] - x_mean[:,np.newaxis]
x_track[:,:,1] = x_track[:,:,1] - y_mean[:,np.newaxis]
x_track[:,:,2] = x_track[:,:,2] - z_mean[:,np.newaxis]

test_track_incorrect[:,:,0] = test_track_incorrect[:,:,0] - x_mean_test[:,np.newaxis]
test_track_incorrect[:,:,1] = test_track_incorrect[:,:,1] - y_mean_test[:,np.newaxis]
test_track_incorrect[:,:,2] = test_track_incorrect[:,:,2] - z_mean_test[:,np.newaxis]

y_track[:,:,0] = y_track[:,:,0] - x_mean[:,np.newaxis]
y_track[:,:,1] = y_track[:,:,1] - y_mean[:,np.newaxis]
y_track[:,:,2] = y_track[:,:,2] - z_mean[:,np.newaxis]

def create_birnn_model(input_shape, hidden_size, output_size):
    model = Sequential()
    model.add(LSTM(hidden_size, input_shape=(input_shape, 3), return_sequences=False))
    # model.add(LSTM(hidden_size, input_shape=(input_shape, 3), return_sequences=True))
    model.add(Flatten())
    model.add(Dense(200))
    # model.add(Dropout(0.2))
    model.add(Dense(30))
    # model.add(Dropout(0.2))
    model.add(Reshape((output_size, 3)))

    # model.add(LSTM(hidden_size, input_shape=(input_shape, 1), return_sequences=True))

    # model.add(SimpleRNN(hidden_size, input_shape=(input_shape, 1), return_sequences=True))
    # model.add(SimpleRNN(hidden_size, input_shape=(input_shape, 1), return_sequences=False))
    # model.add(Dropout(0.2))
    # model.add(Flatten())
    # model.add(Dense(100))
    # model.add(Dense(50))
    # model.add(Dense(output_size))
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
output_size = 10  # 输出维度

# 实例化模型
model = create_birnn_model(input_size, hidden_size, output_size)


optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1)


# 编译模型
model.compile(optimizer=optimizer, loss='mse')

model_dir = "D:\\english\\casic-2\\model\\bi-rnn\\"


my_epochs = 10000

my_start = 70
my_end = 100




# 创建一个包含1到100的列表
lst = list(range(100))
# 随机排序列表
half = len(lst) // 2
random.shuffle(lst)
lst1 = lst[:half]
lst2 = lst[half:]



# model = load_model(model_dir+'model.h5')


history = model.fit(x_track[lst1,:], y_track[lst1,:],validation_data=[x_track[lst2,:],y_track[lst2,:]], epochs=my_epochs, batch_size=100)
# loss_values = history.history['loss']
# # 获取训练过程中的epoch数
# epochs = range(1, len(loss_values) + 1)




model.save(model_dir+'model.h5')

# 前向传播
my_predict = model.predict(x_track)

x_track[:,:,0] = x_track[:,:,0] + x_mean[:,np.newaxis]
x_track[:,:,1] = x_track[:,:,1] + y_mean[:,np.newaxis]
x_track[:,:,2] = x_track[:,:,2] + z_mean[:,np.newaxis]

my_predict[:,0] = my_predict[:,0] + x_mean[:,np.newaxis]
my_predict[:,1] = my_predict[:,1] + y_mean[:,np.newaxis]
my_predict[:,2] = my_predict[:,2] + z_mean[:,np.newaxis]




# predict_x_test = model_x.predict(test_track_incorrect[:,:,0])
# predict_y_test = model_y.predict(test_track_incorrect[:,:,1])
# predict_z_test = model_z.predict(test_track_incorrect[:,:,2])

# test_track_incorrect[:,:,0] = test_track_incorrect[:,:,0] + x_mean_test[:,np.newaxis]
# test_track_incorrect[:,:,1] = test_track_incorrect[:,:,1] + y_mean_test[:,np.newaxis]
# test_track_incorrect[:,:,2] = test_track_incorrect[:,:,2] + z_mean_test[:,np.newaxis]

# predict_x_test = predict_x_test + x_mean_test[:,np.newaxis]
# predict_y_test = predict_y_test + y_mean_test[:,np.newaxis]
# predict_z_test = predict_z_test + z_mean_test[:,np.newaxis]



nums = 10

plt.figure()
plt.rcParams['font.family'] = 'Times New Roman'
ax = plt.axes(projection='3d')

for i in lst2:
    x = x_track[i,:,0]
    y = x_track[i,:, 1]
    z = x_track[i,:, 2]
    ax.plot3D(x,y,z,".",markersize=1,color="blue")

for i in lst2:
    x = my_predict[i,:,0]
    y = my_predict[i,:,1]
    z = my_predict[i,:,2]
    ax.plot3D(x,y,z,".",markersize=1,color="red")

ax.set_xlabel('x / m')
ax.set_ylabel('y / m')
ax.set_zlabel('z / m')

# plt.figure()
# plt.rcParams['font.family'] = 'Times New Roman'
# ax = plt.axes(projection='3d')

# for i in lst2[:1]:
#     x = test_track_incorrect[i,:,0]
#     y = test_track_incorrect[i,:, 1]
#     z = test_track_incorrect[i,:, 2]
#     ax.plot3D(x,y,z,".",markersize=1,color="blue")
#
# for i in lst2[:1]:
#     x = predict_x_test[i,:]
#     y = predict_y_test[i,:]
#     z = predict_z_test[i,:]
#     ax.plot3D(x,y,z,".",markersize=1,color="red")

# ax.set_xlabel('x / m')
# ax.set_ylabel('y / m')
# ax.set_zlabel('z / m')

plt.show()

my_dist = projection_dist()

dist_list = []
for i in lst2:
    dist = my_dist.distance(miss_track[i,:,:],my_predict[i,:,:])
    dist_list.append(dist)
print("dist_list: ",dist_list)
sio.savemat('D:\\实验室\\项目\\二院\\徐老师\\算法\\RNN-补全\\dist_list.mat', {'dist_list': dist_list})