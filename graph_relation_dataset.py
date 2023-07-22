import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data
import h5py
import math
import os.path as osp

base_dir = "D:\\实验室\\项目\\二院\\徐老师\\算法\\100条航迹\\"

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

# 加载.mat文件
mat_file = h5py.File(base_dir+'before_track_matrix.mat', 'r')
# 获取文件中的数据集名称
dataset_name = list(mat_file.keys())[0]
# 获取数据集
dataset = mat_file[dataset_name]
# 将数据集转换为numpy数组
before_track = dataset[()]
# 关闭文件
mat_file.close()

before_track = np.swapaxes(before_track,2,0)
after_track = np.swapaxes(after_track,2,0)

# 这里给出大家注释方便理解
class graph_dataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    # 返回数据集源文件名
    # @property
    # def raw_file_names(self):
    #     return ['some_file_1', 'some_file_2', ...]

    # 返回process方法所需的保存文件名。你之后保存的数据集名字和列表里的一致
    @property
    def processed_file_names(self):
        return ['trainset.pt']

    # def len(self):
    #     return len(self.processed_file_names)
    #
    # def get(self, idx):
    #     data = torch.load(osp.join(self.processed_dir, 'before_data_{}.pt'.format(idx)))
    #     return data

    # 生成数据集所用的方法
    def process(self):
        lst1 = range(1, 99, 2)
        lst2 = range(2, 100, 2)

        data_list = []


        edge_index = np.zeros((2, 276))  # 3
        edge_weight = np.zeros((1, 276))
        count = 0
        for i in range(30):
            for j in range(10):
                a = i - 5 + j
                if a < 0 or a >= 30:
                    continue
                edge_index[0, count] = i
                edge_index[1, count] = a

                count = count + 1
        print("count: ", count)
        Edge_index = torch.tensor(edge_index, dtype=torch.long)


        for i in range(100):
            for j in range(100):
                print("i: ", i,"  j: ", j,)
                if i==j: y = 1
                else: y = 0
                Y = torch.tensor(y, dtype=torch.long)

                track_x = np.concatenate((before_track[i, :, :], after_track[j, :, :]), axis=0)
                X = torch.tensor(track_x, dtype=torch.float)

                count = 0
                for p in range(30):
                    for q in range(10):
                        a = p - 5 + q
                        if a < 0 or a >= 30:
                            continue
                        sum_2 = sum(pow(track_x[p, :]- track_x[q, :],2))
                        weight = pow(0.96, math.sqrt(sum_2)/340)
                        # print("weight: ", weight)
                        edge_weight[0, count] = weight
                        count = count + 1
                Edge_weight = torch.tensor(edge_weight, dtype=torch.float)

                data = Data(x=X, y = Y, edge_index=Edge_index,edge_weight = Edge_weight)
                data_list.append(data)

        # if self.pre_filter is not None:
        #     data_list = [data for data in data_list if self.pre_filter(data)]
        #
        # if self.pre_transform is not None:
        #     data_list = [self.pre_transform(data) for data in data_list]

        data_, slices = self.collate(data_list)#将不同大小的图数据对齐，填充

        #slices可以对填充后的data进行切割
        torch.save((data_, slices), self.processed_paths[0])


my_data = graph_dataset("my_data")
print("  ")