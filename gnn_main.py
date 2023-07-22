import time
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter
import torch_geometric.nn as pyg_nn
from torch_geometric.datasets import Planetoid
import matplotlib.pyplot as plt
import scipy.io as sio
from projection_dist import projection_dist
from graph_trainset import graph_trainset
from graph_valset import graph_valset
from torchviz import make_dot
from torch.nn import Parameter


graph_train_data = graph_trainset("my_data")
graph_val_data = graph_valset("my_data")

class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(GraphConv, self).__init__()
        # 线性层
        self.w1 = pyg_nn.dense.linear.Linear(in_channels, out_channels, weight_initializer='glorot', bias=bias)
        self.w2 = pyg_nn.dense.linear.Linear(in_channels, out_channels, weight_initializer='glorot', bias=bias)
        self.a = Parameter(torch.FloatTensor(1))
        self.b = Parameter(torch.FloatTensor(1))
        self.a.data.fill_(0.5)
        self.b.data.fill_(0.5)

    def forward(self, x, edge_index, edge_weight):
        # 对自身节点进行特征映射
        wh_1 = self.w1(x)



        # 获取邻居特征
        x_j = x[edge_index[0]]

        # 对邻居节点进行聚合
        # x_j = scatter(src=x_j, index=edge_index[1], dim=0, reduce='sum')  # sum聚合操作 [num_nodes, out_channels]

        # x_j = scatter(src=x_j * edge_weight, index=edge_index[1], dim=0, reduce='sum')
        x_j = scatter(src=x_j, index=edge_index[1], dim=0, reduce='sum')

        # 对邻居节点进行特征映射
        wh_2 = self.w2(x_j)

        # print("a: ",self.a.data,"  b: ", self.b.data)

        # wh = self.a*wh_1 + self.b*wh_2
        # wh = wh_1 + wh_2
        # wh = wh_1
        wh = wh_2

        return wh

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = GraphConv(3, 50)
        # self.conv2 = GraphConv(3, 3)
        self.conv2 = GraphConv(53, 50)
        self.conv3 = GraphConv(103, 3)

    def forward(self, data):
        x_old, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

        x_mean = x_old.mean(dim=0)
        # x_max,_ = torch.max(x,dim=0)
        # x_max = x_max.unsqueeze(0)
        x0 = x_old - x_mean
        # x = x/x_max

        x1 = self.conv1(x0, edge_index, edge_weight)
        x1 = torch.cat((x1, x0), dim=1)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x2 = self.conv2(x1, edge_index, edge_weight)
        x2 = torch.cat((x2, x1), dim=1)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x3 = self.conv3(x2, edge_index, edge_weight)
        # x3 = torch.cat((x3, x2), dim=1)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)

        # x = x * x_max
        x_new = x3 + x_mean

        fixed_nodes_before = torch.cat([torch.arange(i * 40, i * 40 + 14) for i in range(49)])
        fixed_nodes_after = torch.cat([torch.arange(i * 40 + 25, i * 40 + 39) for i in range(49)])

        x_new[fixed_nodes_before] = x_old[fixed_nodes_before]
        x_new[fixed_nodes_after] = x_old[fixed_nodes_after]

        return x_new


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
epochs = 1500  # 学习轮数
lr = 0.001 # 学习率 Cora的一张图

# 4.定义模型
model = Model()
# model.to(device)
model = torch.load('D:\english\casic-2\model\gnn\model_predict.pt')
optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 优化器

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9, last_epoch=-1)

# class CustomLoss(nn.Module):
#     def __init__(self):
#         super(CustomLoss, self).__init__()
#
#     def forward(self,score):
#         # 自定义损失计算逻辑
#         return score

loss_function = nn.MSELoss()  # 损失函数
# loss_function = CustomLoss()

# 训练模式
model.train()
loss_list = []
ymax = -1
# plt.figure()
# plt.ion()
graph_data_y_flatten = torch.flatten(graph_train_data.y)

for epoch in range(epochs):
    # if epoch!=0:
    optimizer.zero_grad()  # 清零优化器梯度，梯度不清零会一直存在
    pred = model(graph_train_data)
    pred_flatten = torch.flatten(pred)
    loss = loss_function(pred_flatten, graph_data_y_flatten)  # 计算一次损失
    #loss反向传播就行，这里没有acc监视器
    loss.backward()
    print("epoch: ", epoch, " loss: ", loss.item())
    #用反向传播得到的梯度进行参数更新
    optimizer.step()

    if loss.item()>ymax:
        ymax = loss.item()
    loss_list.append(loss.item())
    x = np.linspace(1,len(loss_list),len(loss_list))
    plt.cla()
    plt.plot(x,loss_list,'.')
    plt.pause(0.2)
    if len(loss_list)<10:
        plt.xlim(1,10)
    else:
        plt.xlim(1, len(loss_list)+1)
    plt.ylim(0, ymax)

torch.save(model, 'D:\english\casic-2\model\gnn\model_predict.pt')

pred = model(graph_val_data)

print('【Finished Training！】')

plt.ioff()

plt.figure()
plt.rcParams['font.family'] = 'Times New Roman'
ax = plt.axes(projection='3d')

my_start = 40
my_end = 49
nums = 10

miss_track = np.zeros((50,10,3))
for i in range(my_start,my_end):
    x = graph_val_data.y[i*40:(i+1)*40, 0].numpy()
    y = graph_val_data.y[i*40:(i+1)*40, 1].numpy()
    z = graph_val_data.y[i*40:(i+1)*40, 2].numpy()
    miss_track[i, :, :] = np.stack((x[15:25], y[15:25], z[15:25]), axis=1)
    ax.plot3D(x,y,z,".",markersize=1,color="blue")


prepare_track = np.zeros((50,10,3))
for i in range(my_start,my_end):
    x = graph_val_data.x[i*40:(i+1)*40, 0].numpy()
    y = graph_val_data.x[i*40:(i+1)*40, 1].numpy()
    z = graph_val_data.x[i*40:(i+1)*40, 2].numpy()
    prepare_track[i, :, :] = np.stack((x[15:25], y[15:25], z[15:25]), axis=1)

ax.set_xlabel('x / m')
ax.set_ylabel('y / m')
ax.set_zlabel('z / m')


plt.figure()
plt.rcParams['font.family'] = 'Times New Roman'
ax = plt.axes(projection='3d')

predict_T = np.zeros((50,10,3))
for i in range(my_start,my_end):
    x = pred[i*40:(i+1)*40, 0].detach().numpy()
    y = pred[i*40:(i+1)*40, 1].detach().numpy()
    z = pred[i*40:(i+1)*40, 2].detach().numpy()
    predict_T[i,:,:] = np.stack((x[15:25],y[15:25],z[15:25]),axis=1)
    ax.plot3D(x,y,z,".",markersize=1,color="blue")


# for i in range(my_start,my_end):
#     x = graph_data.get(i).x[:,0]
#     y = graph_data.get(i).x[:,1]
#     z = graph_data.get(i).x[:,2]
#     ax.plot3D(x,y,z,".",markersize=1,color="blue")


ax.set_xlabel('x / m')
ax.set_ylabel('y / m')
ax.set_zlabel('z / m')

plt.show()





my_dist = projection_dist()


dist_list = []
for i in range(50):
    dist = my_dist.distance(miss_track[i,:,:],predict_T[i,:,:])
    dist_list.append(dist)
print("dist_list: ",dist_list)
print(np.mean(dist_list))

# 测试图的补全能力是否和未映射的补长节点有关
dist_list_2 = []
for i in range(50):
    dist = my_dist.distance(miss_track[i,:,:],prepare_track[i,:,:])
    dist_list_2.append(dist)
print("dist_list_2: ",dist_list_2)
print(np.mean(dist_list_2))

sio.savemat('D:\\实验室\\项目\\二院\\徐老师\\算法\\graph-补全\\dist_list.mat', {'dist_list': dist_list})

