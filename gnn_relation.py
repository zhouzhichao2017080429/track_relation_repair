import time
import h5py
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

from torchviz import make_dot
from torch.nn import Linear,Sigmoid,LeakyReLU
from torch_geometric.utils import scatter,add_self_loops, degree
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import global_mean_pool,MessagePassing
from torch_geometric.data import Batch

from graph_dataset import graph_dataset




model_dir = "D:\\english\\casic-2\\model\\gnn-relation\\"
graph_data = graph_dataset("my_data")

# graph_data.data.edge_weight = graph_data.data.edge_weight.reshape(1, graph_data.data.edge_weight.shape[0]*graph_data.data.edge_weight.shape[1])

data_1 = graph_data[graph_data.y == 1]
data_0 = graph_data[graph_data.y == 0]
data_0 = data_0[:500]

# graph_data = Batch.from_data_list(data_0+data_1)
graph_data = Batch.from_data_list(data_0+data_1)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for i in range(75,85):
#     before_track = before_track_data.get(i)
#     after_track = after_track_data.get(i)
#
#     ax.scatter(before_track.x[:, 0].tolist(), before_track.x[:, 1].tolist(), before_track.x[:, 2].tolist(), c='b',
#                marker='.')
#     ax.scatter(after_track.x[:, 0].tolist(), after_track.x[:, 1].tolist(), after_track.x[:, 2].tolist(),
#                c='r',
#                marker='.')
# plt.show()

class GATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_heads):
        super(GATConv, self).__init__(aggr='add')
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads

        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.att = nn.Parameter(torch.Tensor(1, num_heads, 2 * self.head_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index):
        x = self.linear(x)

        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)#节点的度
        deg_inv_sqrt = deg.pow(-0.5)#节点的度倒数的平方根
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]#用于后面softmax归一化

        x = self.propagate(edge_index, x=x, norm=norm)
        return x

    def message(self, x_j, norm):
        alpha = (x_j * self.att[:, :, :self.head_dim]).sum(dim=-1)
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        # alpha2 = torch.softmax(alpha, norm)
        alpha = torch.softmax(alpha, dim=-1)
        return x_j * alpha.unsqueeze(-1)
        # return x_j * alpha

    def update(self, aggr_out):
        return aggr_out


class GINConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GINConv, self).__init__(aggr='add')
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 10),
            nn.ReLU(),
            nn.Linear(10, out_channels),
            nn.ReLU()
        )
        self.eps = nn.Parameter(torch.Tensor([0]))

    def forward(self, x, edge_index):
        # edge_index2, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x))
        return x

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out


class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(GraphConv, self).__init__()
        # 线性层
        self.w1 = pyg_nn.dense.linear.Linear(in_channels, out_channels, weight_initializer='glorot', bias=bias)
        self.w2 = pyg_nn.dense.linear.Linear(in_channels, out_channels, weight_initializer='glorot', bias=bias)

    def forward(self, x, edge_index, edge_weight):
        # 对自身节点进行特征映射
        wh_1 = self.w1(x)

        # 获取邻居特征
        x_j = x[edge_index[0]]
        #
        # # 对邻居节点进行聚合
        # x_j = scatter(src=x_j, index=edge_index[1], dim=0, reduce='sum')  # sum聚合操作 [num_nodes, out_channels]
        # edge_weight = torch.transpose(edge_weight, 0, 1)


        # # 重塑成1x165600大小的张量
        # edge_weight = edge_weight.reshape(1, edge_weight.shape[0]*edge_weight.shape[1])
        edge_weight = edge_weight.reshape(edge_weight.shape[0] * edge_weight.shape[1],1)

        x_j = scatter(src=x_j * edge_weight, index=edge_index[1], dim=0, reduce='sum')
        # x_j = scatter(src=x_j, index=edge_index[1], dim=0, reduce='sum')
        #
        # # 对邻居节点进行特征映射
        wh_2 = self.w2(x_j)

        # wh = wh_1 + wh_2
        # wh = wh_1
        wh = wh_2


        return wh

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.gcn1 = GraphConv(3, 32)
        # self.gcn1 = GraphConv(3, 10)
        # self.conv2 = GraphConv(3, 3)
        self.gcn2 = GraphConv(64, 64)
        self.gcn3 = GraphConv(128, 128)
        self.gcn4 = GraphConv(256, 256)
        # self.gcn5 = GraphConv(140, 20)

        self.gin1 = GINConv(3, 32)
        self.gin2 = GINConv(64, 64)
        self.gin3 = GINConv(128, 128)
        self.gin4 = GINConv(256, 256)
        # self.gin5 = GINConv(3, 64)

        self.gat = GATConv(256, 1,1)


        self.leak_relu1 = LeakyReLU(negative_slope=0.2)
        self.leak_relu2 = LeakyReLU(negative_slope=0.2)
        self.leak_relu3 = LeakyReLU(negative_slope=0.2)
        self.leak_relu4 = LeakyReLU(negative_slope=0.2)
        self.leak_relu5 = LeakyReLU(negative_slope=0.2)

        self.lin = Linear(256, 10, bias=True)
        self.lin2 = Linear(10, 1, bias=True)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        n = int(x.shape[0] / 30)
        x0 = x - x.mean(dim=0)
        # x0 = x

        # x0 = x0.view(n, 30 * 3)
        # x0 = x0.reshape(n, 30, 3)
        # x0 = x0.permute(0, 2, 1)
        # x0 = x0.reshape(n, -1)



        x0 = x0/x0.max(dim=0).values

        # x3 = x0

        x11 = self.gcn1(x0, edge_index, edge_weight)
        x12 = self.gin1(x0, edge_index)
        x1 = torch.cat((x11, x12), dim=1)
        x1 = self.leak_relu1(x1)

        # x1 = F.dropout(x1, training=self.training)
        x21 = self.gcn2(x1, edge_index, edge_weight)
        x22 = self.gin2(x1, edge_index)
        x2 = torch.cat((x21, x22), dim=1)
        x2 = self.leak_relu2(x2)

        # x2 = F.dropout(x2, training=self.training)
        x31 = self.gcn3(x2, edge_index, edge_weight)
        x32 = self.gin3(x2, edge_index)
        x3 = torch.cat((x31, x32), dim=1)
        x3 = self.leak_relu3(x3)
        #
        # x41 = self.gcn4(x3, edge_index, edge_weight)
        # x42 = self.gin4(x3, edge_index)
        # x4 = torch.cat((x41, x42), dim=1)
        # x4 = self.leak_relu4(x4)

        # x5 = self.gcn5(x4, edge_index, edge_weight)
        # x5 = torch.cat((x5, x4), dim=1)
        # x5 = self.leak_relu5(x5)

        att_weight = self.gat(x3, edge_index)
        att_weight = att_weight.squeeze()
        att_weight = att_weight.unsqueeze(-1)


        batch_np  = np.repeat(np.arange(n), 30)#用于做全局池化，在样本数量和节点数上都做全局池化
        batch = torch.tensor(batch_np , dtype=torch.long)

        x5 = global_mean_pool(x3*att_weight, batch)
        # x5 = global_mean_pool(x3, batch)

        f_length = 100

        # x4 = x3.view(n, 30 * f_length)
        #
        # x4 = x4.reshape(n, 10 * f_length, 3)
        # x4 = x4.permute(0, 2, 1)
        # x4 = x4.reshape(n, -1)

        x5 = self.lin(x5)
        # x5 = F.relu(x5)
        x6 = self.lin2(x5)
        # x6 = F.relu(x6)

        # x6 = self.lin3(x6)
        x6 = F.sigmoid(x6)
        x6 = x6.squeeze(1)
        return x6


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
epochs = 500  # 学习轮数
lr = 0.003 # 学习率 Cora的一张图

# 4.定义模型
model = Model()
# model.to(device)
# model = torch.load(model_dir+'model.pt')
optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 优化器

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9, last_epoch=-1)

# class CustomLoss(nn.Module):
#     def __init__(self):
#         super(CustomLoss, self).__init__()
#
#     def forward(self,score):
#         # 自定义损失计算逻辑
#         return score

# loss_function = nn.NLLLoss()
loss_function = nn.BCELoss() # 损失函数
# loss_function = nn.BCEWithLogitsLoss()
# loss_function = CustomLoss()


# 训练模式
model.train()
plt.figure()
plt.ion()
loss_list = []
score_p_list = []
score_n_list = []
ymax = -1

for epoch in range(epochs):


    # if epoch!=0:
    optimizer.zero_grad()  # 清零优化器梯度，梯度不清零会一直存在

    # score = score.to(device)
    correct_count = 0

    # pred = model(before_track_data.get(p).to(device), after_track_data.get(q).to(device))

    pre = model(graph_data)
    pre_1 = model(data_1)
    pre_0 = model(data_0)


    loss = loss_function(pre.float(), graph_data.y.float())  # 计算一次损失
    # loss = loss_function(pre_1.float(), data_1.y.float())
    # loss = loss_function(pre_1, data_1.y)


    #loss反向传播就行，这里没有acc监视器
    loss.backward()



    # print("pre: ", pre[:20].detach().numpy())
    print("pre_1: ", pre_1.detach().long()[:30].numpy())
    print("pre_0: ", pre_0.detach().long()[:30].numpy())
    # print("data_1.y: ", data_1.y.numpy())

    mask = torch.lt(torch.abs(pre_1 - 1), 0.1)

    # 计算True元素的数量
    count = torch.sum(mask).numpy()


    acc = count/100
    print("epoch: ", epoch,"  loss: ", loss.item(), "  acc: ", acc)

    print(" ")

    #用反向传播得到的梯度进行参数更新
    optimizer.step()

torch.save(model, model_dir+'model.pt')


pre = model(graph_data)
pre_1 = model(data_1)
pre_0 = model(data_0)
mask = torch.lt(torch.abs(pre_1 - 1), 0.1)
# 计算True元素的数量
count = torch.sum(mask).numpy()
acc = count/100
print("  acc: ", acc)


print('【Finished Training！】')





