import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data
import h5py
import math
import os.path as osp
from playsound import playsound

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
# x_track = before_track
y_track = miss_track.copy()

_y_track = np.zeros(y_track.shape)
for i in range(100):
    last_point = before_track[i,-1,:]
    first_point = after_track[i,0,:]
    track = np.linspace(last_point,first_point,10)
    # track = np.linspace(last_point, last_point, 10)
    _y_track[i,:,:] = track


# 这里给出大家注释方便理解
class graph_valset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    # 返回process方法所需的保存文件名。你之后保存的数据集名字和列表里的一致
    @property
    def processed_file_names(self):
        return ['valset.pt']

    # 生成数据集所用的方法
    def process(self):
        track_num = 100
        lst1 = range(1, 99, 2)
        lst2 = range(2, 100, 2)

        # indices = list(range(track_num))
        #
        # # 打乱索引顺序
        # random.shuffle(indices)
        # train_size = 50
        # train_indices = indices[:train_size]
        # val_indices = indices[train_size:]
        # train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        # train_mask[:80] = True

        data_list = []

        edge_index = np.zeros((2, 119))  # 3
        # edge_index = np.zeros((2, 195))#5
        # edge_index = np.zeros((2, 732))#21
        edge_weight = np.zeros((1, 119))
        count = 0
        for i in range(40):
            for j in range(3):
                a = i-1+j
                if a<0 or a>=40:
                    continue
                edge_index[0, count] = i
                edge_index[1, count] = a
                # edge_weight[0,count] = (10 - np.abs(j - 10)) / 10
                count = count + 1
        print("count: ",count)
        Edge_index = torch.tensor(edge_index, dtype=torch.long)
        Edge_weight = torch.tensor(edge_weight, dtype=torch.float)

        for i in lst2:
            track_x = np.concatenate((before_track[i, :, :],_y_track[i,:,:],after_track[i, :, :]),axis=0)
            track_y = np.concatenate((before_track[i, :, :], y_track[i, :, :], after_track[i, :, :]), axis=0)


            X = torch.tensor(track_x, dtype=torch.float)
            Y = torch.tensor(track_y, dtype=torch.float)


            # Edge_index = torch.ones(2, 39, dtype=torch.long)

            data = Data(x=X, edge_index=Edge_index, edge_weight=Edge_weight, y=Y)

            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]


        data_, slices = self.collate(data_list)#将不同大小的图数据对齐，填充

        #slices可以对填充后的data进行切割
        torch.save((data_, slices), self.processed_paths[0])



my_data = graph_valset("my_data")
print("...数据集生成完成...")