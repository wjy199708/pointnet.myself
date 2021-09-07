from os import O_ASYNC, truncate
from typing import overload
from mmdet import models
import torch
from torch._C import set_flush_denormal
from torch.nn import Module
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.linear import Linear


class InputTransform(Module):
    def __init__(self):
        super(InputTransform, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)

        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        #  the last layer linear out
        self.fc3 = nn.Linear(256, 9)

    def forward(self, x):
        bs = x.size()[0]

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        # print(x.size())

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))

        x = self.fc3(x)

        #  根据 论文提示需要将   单位矩阵  作为输出矩阵的初始化矩阵
        identity = torch.eye(1, 9).repeat(bs, 1)

        x = x + identity

        x = x.view(-1, 3, 3)
        return x


class Feature_Trans(Module):
    def __init__(self, k=64):
        super(Feature_Trans, self).__init__()
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, k * k)
        self.k = k

    def forward(self, x):
        ''' x (b,64,n) '''
        bs = x.size()[0]
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))  # (b,1024,n)

        x = torch.max_pool1d(x, x.size()[2])
        # print(x.size())
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))

        x = self.fc3(x)

        identity = torch.eye(self.k, self.k).flatten()

        x = x + identity
        x = x.view(-1, self.k, self.k)
        return x


class PointNet(Module):
    def __init__(self, is_feature_trans=False, gloab_feature=True):
        super(PointNet, self).__init__()
        self.input_trans = InputTransform()

        self.mlp1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)

        self.is_feature_trans = is_feature_trans
        if is_feature_trans:
            self.feature_trans = Feature_Trans()

        self.mlp2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)

        self.mlp3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        ''' x : (b,n,3) 

        '''
        #  计算  输入转换矩阵
        num_points = x.size()[1]
        x = torch.Tensor.permute(x, 0, 2, 1)
        input_trasforme = self.input_trans(x)  # (-1,3,3)
        x = torch.Tensor.transpose(x, 2, 1)
        # x = x@input_trasforme
        x = torch.bmm(x, input_trasforme)

        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.mlp1(x)))  # (b,64,n)

        if self.is_feature_trans:
            feature_trans = self.feature_trans(x)  # (b,64,64)
            x = torch.Tensor.permute(x, 0, 2, 1)
            x = torch.bmm(x, feature_trans)
            x = torch.Tensor.permute(x, 0, 2, 1)
        else:
            trans_feat = None
        # print(x.size())

        x = F.relu(self.bn2(self.mlp2(x)))
        x = self.bn3(self.mlp3(x))  # (b,1024,n)

        x = torch.max_pool1d(x, num_points)

        x = x.view(-1, 1024)

        ''' 全局 特征提取 '''
        # global_feature=torch.cat([])
        return x


if __name__ == '__main__':
    ptc = torch.rand(3, 16128, 3)
    # ptc = ptc.permute(0, 2, 1)
    # model_input_trans = InputTransform()
    # out = model_input_trans(ptc)
    # print(out)

    # feature_trans=Feature_Trans()
    # x=torch.rand(3,64,16800)
    # out=feature_trans(x)
    # print(out.size())

    pointnet = PointNet(is_feature_trans=True)
    out = pointnet(ptc)
    print(out.size())
