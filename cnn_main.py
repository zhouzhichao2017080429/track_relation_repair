#-*- coding : utf-8-*-

import h5py
import math
import numpy as np
from keras.models import Sequential,load_model,save_model
from tensorflow.keras.callbacks import EarlyStopping
from keras.layers import Conv1D, Dense, Flatten,MaxPooling1D,Reshape
import torch
from numpy import random
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from keras import backend as K
from torch_geometric.data import InMemoryDataset, Data
import matplotlib.pyplot as plt
import os.path as osp
from projection_dist import projection_dist
import scipy.io as sio
from tensorflow.keras.optimizers import Adam


# tf.compat.v1.disable_eager_execution()


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
y_track = miss_track.copy()

x_mean = np.mean(x_track[:,:,0],axis=1)
y_mean = np.mean(x_track[:,:,1],axis=1)
z_mean = np.mean(x_track[:,:,2],axis=1)

x_max = np.max(x_track[:,:,0],axis=1)
y_max = np.max(x_track[:,:,1],axis=1)
z_max = np.max(x_track[:,:,2],axis=1)

x_track[:,:,0] = x_track[:,:,0] - x_mean[:,np.newaxis]
x_track[:,:,1] = x_track[:,:,1] - y_mean[:,np.newaxis]
x_track[:,:,2] = x_track[:,:,2] - z_mean[:,np.newaxis]

# x_track[:,:,0] = x_track[:,:,0]/x_max[:,np.newaxis]
# x_track[:,:,1] = x_track[:,:,1]/y_max[:,np.newaxis]
# x_track[:,:,2] = x_track[:,:,2]/z_max[:,np.newaxis]





x_mean_test = np.mean(test_track_incorrect[:,:,0],axis=1)
y_mean_test = np.mean(test_track_incorrect[:,:,1],axis=1)
z_mean_test = np.mean(test_track_incorrect[:,:,2],axis=1)

x_max_test = np.max(test_track_incorrect[:,:,0],axis=1)
y_max_test = np.max(test_track_incorrect[:,:,1],axis=1)
z_max_test = np.max(test_track_incorrect[:,:,2],axis=1)

test_track_incorrect[:,:,0] = test_track_incorrect[:,:,0] - x_mean_test[:,np.newaxis]
test_track_incorrect[:,:,1] = test_track_incorrect[:,:,1] - y_mean_test[:,np.newaxis]
test_track_incorrect[:,:,2] = test_track_incorrect[:,:,2] - z_mean_test[:,np.newaxis]

y_track[:,:,0] = y_track[:,:,0] - x_mean[:,np.newaxis]
y_track[:,:,1] = y_track[:,:,1] - y_mean[:,np.newaxis]
y_track[:,:,2] = y_track[:,:,2] - z_mean[:,np.newaxis]

# y_track[:,:,0] = y_track[:,:,0]/x_max[:,np.newaxis]
# y_track[:,:,1] = y_track[:,:,1]/y_max[:,np.newaxis]
# y_track[:,:,2] = y_track[:,:,2]/z_max[:,np.newaxis]


nums = 5


# 创建模型
def cnn_model():
    my_model = Sequential()
    my_model.add(Conv1D(3, 3, input_shape=(30, 3)))
    my_model.add(MaxPooling1D(pool_size=2))
    my_model.add(Conv1D(3, 3))
    my_model.add(Conv1D(3, 3))
    my_model.add(Flatten())
    my_model.add(Dense(30))
    my_model.add(Reshape((10, 3)))

    # my_model.add(Dense(10))
    # my_model.add(Dense(10))
    # my_model.add(Dense(10))
    return my_model

model = cnn_model()

optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)


# def myself_loss(truth_T,predicted_T):
#     dist = K.mean(K.square(truth_T - predicted_T), axis=-1)
#     min_dist_list = tf.zeros((1, 1), dtype=tf.float32)
#     for i in range(len(truth_T)):
#         min_dist = 1e6
#         for k in range(len(predicted_T)):
#             dist = tf.sqrt(tf.reduce_sum((truth_T[i, :] - predicted_T[k, :]) ** 2))
#             if dist < min_dist:
#                 min_dist = dist
#
#         min_dist = tf.cast(min_dist, tf.float32)
#         min_dist_list = min_dist_list + min_dist
#     dist = min_dist_list / len(truth_T)
#     print("dist: ", dist.numpy())
#     return dist


# def myself_loss(truth_T,predicted_T):
#     dist = K.mean(K.square(truth_T - predicted_T), axis=-1)
#     return dist

def myself_loss(truth_T_list,predicted_T_list):
    # dist_sum = tf.zeros((1,1))
    # for i in range(len(truth_T_list)):
    #     #真值是一行摆开
    #     dist_matrix = tf.abs(truth_T_list[i,:] - predicted_T_list[i,:])
    #     my_mean = tf.reduce_mean(dist_matrix, axis=0)
    #     dist_sum = dist_sum + my_mean
    # dist_all = dist_sum/len(truth_T_list)

    # dist_all = truth_T_list - predicted_T_list
    # dist_all = tf.abs(truth_T_list - predicted_T_list)
    # dist_all = tf.abs(truth_T_list - predicted_T_list)*tf.abs(truth_T_list - predicted_T_list)
    dist_all = tf.square(truth_T_list - predicted_T_list)
    # dist_all = tf.square(truth_T_list - predicted_T_list)
    # dist_all = tf.abs(truth_T_list - predicted_T_list)*tf.abs(truth_T_list - predicted_T_list)*tf.abs(truth_T_list - predicted_T_list)/1000000000000
    # dist_all = (truth_T_list - predicted_T_list) * (truth_T_list - predicted_T_list) * (truth_T_list - predicted_T_list) / 1000000000000
    # dist_all = tf.square(truth_T_list - predicted_T_list)*tf.square(truth_T_list - predicted_T_list) / 1000000000000
    # dist_all = tf.reduce_mean(dist_all, axis=-1)
    # dist_all = tf.sqrt(dist_all)
    dist_all = dist_all
    return dist_all


# def myself_loss(truth_T_list,predicted_T_list):
#     # dist_list = []
#     dist_sum = tf.zeros((1,1))
#     for i in range(len(truth_T_list)):
#         #真值是一行摆开
#         dist_matrix = tf.square(truth_T_list[i,:,np.newaxis] - predicted_T_list[i,np.newaxis,:])
#         # dist_matrix = tf.abs(truth_T_list[i, :] - predicted_T_list[i, :])
#         # print("dist_matrix: \n", dist_matrix)
#         # my_min = tf.reduce_min(dist_matrix, axis=0)
#         my_min = tf.reduce_min(dist_matrix, axis=1)
#         # print("my_min: \n", my_min)
#         dist = tf.reduce_mean(my_min)
#         # dist_list.append(dist)
#         dist_sum = dist_sum + dist
#
#
#     # dist_all = tf.reduce_mean(dist_list)
#
#     dist_all = dist_sum/len(truth_T_list)
#
#     # dist_list = tf.convert_to_tensor(dist_list)
#
#     # print("dist: ", dist)
#     # print(" ")
#     return dist_all


# def myself_loss(truth_T, predicted_T):
#     min_dist_list = np.zeros((len(truth_T), 1))
#     for i in range(len(truth_T)):
#         min_dist = 1e6
#         for k in range(len(predicted_T)):
#             dist = np.sqrt(sum((truth_T[i, :] - predicted_T[k, :]) ** 2))
#             if dist < min_dist:
#                 min_dist = dist
#
#         min_dist_list[i, 0] = min_dist
#     dist = np.sum(min_dist_list[:]) / len(truth_T)
#     return dist



model.compile(optimizer=optimizer, loss=myself_loss)
# model_x.compile(optimizer=optimizer_x, loss='mse')
model_dir = "D:\\english\\casic-2\\model\\cnn\\"


# 训练模型
my_epochs = 1000

# from tensorflow.keras.callbacks import LambdaCallback
# loss_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: print(f"Epoch {epoch+1} - Loss: {logs['loss']}"))


early_stopping = EarlyStopping(monitor='loss', patience=100)
# model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

# model = load_model(model_dir+'model.h5', custom_objects={'myself_loss': myself_loss})


# 创建一个包含1到100的列表
lst = list(range(100))
# 随机排序列表
half = len(lst) // 2
random.shuffle(lst)
lst1 = lst[:half]
lst2 = lst[half:]




history = model.fit(x_track[lst1,:], y_track[lst1,:],validation_data=[x_track[lst2,:], y_track[lst2,:]], epochs=my_epochs, batch_size=100)
model.save(model_dir+'model.h5')

# loss_values = history.history['val_loss']
# loss_values = loss_values[10:]
# # 获取训练过程中的epoch数
# epochs = range(1, len(loss_values) + 1)
# plt.plot(epochs,loss_values)
# plt.xlabel("epochs")
# plt.ylabel("loss")
# plt.grid(True)



predict = model.predict(x_track)




nums = 10
my_start = 0
my_end = 100


x_track[:,:,0] = x_track[:,:,0] + x_mean[:,np.newaxis]
x_track[:,:,1] = x_track[:,:,1] + y_mean[:,np.newaxis]
x_track[:,:,2] = x_track[:,:,2] + z_mean[:,np.newaxis]




# predict[:,:,0] = predict[:,:,0]*x_max[:,np.newaxis]
# predict[:,:,1] = predict[:,:,1]*y_max[:,np.newaxis]
# predict[:,:,2] = predict[:,:,2]*z_max[:,np.newaxis]

predict[:,:,0] = predict[:,:,0] + x_mean[:,np.newaxis]
predict[:,:,1] = predict[:,:,1] + y_mean[:,np.newaxis]
predict[:,:,2] = predict[:,:,2] + z_mean[:,np.newaxis]



# test_track_incorrect[:,:,0] = test_tra ck_incorrect[:,:,0] + x_mean_test[:,np.newaxis]
# test_track_incorrect[:,:,1] = test_track_incorrect[:,:,1] + y_mean_test[:,np.newaxis]
# test_track_incorrect[:,:,2] = test_track_incorrect[:,:,2] + z_mean_test[:,np.newaxis]
#
# predict_x_test = predict_x_test + x_mean_test[:,np.newaxis]
# predict_y_test = predict_y_test + y_mean_test[:,np.newaxis]
# predict_z_test = predict_z_test + z_mean_test[:,np.newaxis]


plt.figure()
plt.rcParams['font.family'] = 'Times New Roman'
ax = plt.axes(projection='3d')

for i in lst2:
    x = x_track[i,:,0]
    y = x_track[i,:, 1]
    z = x_track[i,:, 2]
    ax.plot3D(x,y,z,".",markersize=1,color="blue")

for i in lst2:
    x = predict[i,:,0]
    y = predict[i,:,1]
    z = predict[i,:,2]
    ax.plot3D(x,y,z,".",markersize=1,color="red")

ax.set_xlabel('x / m')
ax.set_ylabel('y / m')
ax.set_zlabel('z / m')
#
# plt.figure()
# plt.rcParams['font.family'] = 'Times New Roman'
# ax = plt.axes(projection='3d')
#
# for i in lst2[:5]:
#     x = test_track_incorrect[i,:,0]
#     y = test_track_incorrect[i,:, 1]
#     z = test_track_incorrect[i,:, 2]
#     ax.plot3D(x,y,z,".",markersize=1,color="blue")
#
# for i in lst2[:5]:
#     x = predict_x_test[i,:]
#     y = predict_y_test[i,:]
#     z = predict_z_test[i,:]
#     ax.plot3D(x,y,z,".",markersize=1,color="red")
#
# ax.set_xlabel('x / m')
# ax.set_ylabel('y / m')
# ax.set_zlabel('z / m')

plt.show()

my_dist = projection_dist()

dist_list = []
# for i in range(100):
for i in lst2:
    dist = my_dist.distance(miss_track[i,:,:],predict[i,:,:])
    dist_list.append(dist)
print("dist_list: ",dist_list)
sio.savemat('D:\\实验室\\项目\\二院\\徐老师\\算法\\CNN-补全\\dist_list.mat', {'dist_list': dist_list})

