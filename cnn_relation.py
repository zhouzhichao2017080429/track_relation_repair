#-*- coding : utf-8-*-

import h5py
import math
import numpy as np
from keras.models import Sequential,load_model,save_model
from tensorflow.keras.callbacks import EarlyStopping
from keras.layers import Conv1D, Dense, Flatten,MaxPooling1D,Conv2D,MaxPooling2D,Dropout
import torch
from numpy import random
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from keras import backend as K
from torch_geometric.data import InMemoryDataset, Data
import matplotlib.pyplot as plt
import os.path as osp
import scipy.io as sio
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l1, l2

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

x_track_0[:,:,0] = x_track_0[:,:,0]/x_max_0[:,np.newaxis]
x_track_0[:,:,1] = x_track_0[:,:,1]/y_max_0[:,np.newaxis]
x_track_0[:,:,2] = x_track_0[:,:,2]/z_max_0[:,np.newaxis]

# y_1 = np.ones((len(x_track_1),1))
# y_0 = np.zeros((len(x_track_0[:500]),1))
# x_track = np.concatenate((x_track_1,x_track_0[:500]),axis=0)


y_1 = np.ones((len(x_track_1[lst1]),1))
y_0 = np.zeros((len(x_track_0[:500]),1))
x_track = np.concatenate((x_track_1[lst1],x_track_0[:500]),axis=0)



# y_1 = np.ones((len(x_track_1),1))
# y_0 = np.zeros((len(x_track_0),1))
# x_track = np.concatenate((x_track_1,x_track_0),axis=0)


y_label = np.concatenate((y_1,y_0),axis=0)



nums = 5


# 创建模型
def cnn_model():
    my_model = Sequential()
    my_model.add(Conv1D(30, 3, input_shape=(30, 3)))#, kernel_regularizer=l2(0.01)
    # my_model.add(Conv2D(input_shape=(30, 3,1), filters=16, kernel_size=(3, 1), padding='same', activation='relu'))
    # my_model.add(MaxPooling2D(pool_size=(3, 1), padding='same'))  # 最大池化


    my_model.add(MaxPooling1D(pool_size=4))#, kernel_regularizer=l2(0.01)
    # my_model.add(Dropout(0.1))

    # my_model.add(Conv1D(30, 3))
    # my_model.add(MaxPooling1D(pool_size=2))
    # my_model.add(Conv1D(30, 3))
    # my_model.add(Conv1D(30, 3))
    # my_model.add(Conv1D(30, 3))
    # my_model.add(MaxPooling1D(pool_size=2))
    # my_model.add(Dense(20, input_shape=(30, )))
    my_model.add(Flatten())
    my_model.add(Dense(100))
    my_model.add(Dropout(0.1))
    my_model.add(Dense(10))#, kernel_regularizer=l2(0.01)
    # my_model.add(Dropout(0.1))
    # my_model.add(Dense(10, activation='relu'))  # , kernel_regularizer=l2(0.01)

    my_model.add(Dense(1, activation='sigmoid'))
    return my_model

model = cnn_model()

print(model.summary())

# 编译模型

optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)


model.compile(optimizer=optimizer, loss='binary_crossentropy')


model_dir = "D:\\english\\casic-2\\model\\cnn-relation\\"




# 训练模型
my_epochs = 10000

early_stopping = EarlyStopping(monitor='loss', patience=100)
# model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

model = load_model(model_dir+'model_relation.h5')


# history = model.fit(x_track, y_label,validation_data=[x_track_1[lst2],np.ones((len(lst2),1))], epochs=my_epochs, batch_size=1000)
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

x_track_1[:,:,0] = x_track_1[:,:,0]*x_max_1[:,np.newaxis]
x_track_1[:,:,1] = x_track_1[:,:,1]*y_max_1[:,np.newaxis]
x_track_1[:,:,2] = x_track_1[:,:,2]*z_max_1[:,np.newaxis]

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
    print("a: ",a)
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


