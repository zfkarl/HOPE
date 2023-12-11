# from models.MVCNN import MVCNN
from __future__ import division, absolute_import
from models.resnet import resnet18
from tools.utils import calculate_accuracy
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import argparse
import torch.optim as optim
import time

# from .Model import Model

class SingleViewNet(nn.Module):

    def __init__(self, pre_trained = None):
        super(SingleViewNet, self).__init__()

        if pre_trained:
            self.img_net = torch.load(pre_trained)
        else:
            print('---------Loading ImageNet pretrained weights --------- ')
            resnet18 = models.resnet18(pretrained=True)
            resnet18 = list(resnet18.children())[:-1]
            self.img_net = nn.Sequential(*resnet18)
            self.linear1 = nn.Linear(512, 256, bias=False)
            self.bn6 = nn.BatchNorm1d(256)

    def forward(self, img, img_v):

        img_feat = self.img_net(img)
        img_feat_v = self.img_net(img_v)
        img_feat = img_feat.squeeze(3)
        img_feat = img_feat.squeeze(2)
        img_feat_v = img_feat_v.squeeze(3)
        img_feat_v = img_feat_v.squeeze(2)

        img_feat = F.relu(self.bn6(self.linear1(img_feat)))
        img_feat_v = F.relu(self.bn6(self.linear1(img_feat_v)))

        final_feat = 0.5*(img_feat + img_feat_v)
        #final_feat = F.normalize(final_feat,1)
        
        return final_feat

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)

    return feature

class DGCNN(nn.Module):
    def __init__(self,  output_channels=10):
        super(DGCNN, self).__init__()
        self.k = 20

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(512)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 512, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(512*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.4)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.4)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        concat = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(concat)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = F.relu(self.bn7(self.linear2(x)))
        #x = F.normalize(x,1)
        
        return x

# class CorrNet(nn.Module):

#     def __init__(self, num_classes):
#         super(CorrNet, self).__init__()
#         self.num_classes=num_classes
#         self.img_net = SingleViewNet(pre_trained=None)
#         self.pt_net = DGCNN(self.num_classes)
# 	#shared head for all feature encoders
#         self.head = nn.Sequential(*[nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, self.num_classes)])

#     def forward(self, img, img_v, pt):

# 	#extract image features
#         img_feat = self.img_net(img, img_v)

# 	#extract mesh features
#         pt_feat = self.pt_net(pt)

#         cmb_feat = (img_feat  + pt_feat)/2.0

# 	#the classification predictions based on image features
#         img_pred = self.head(img_feat)

# 	#the classification prediction based on mesh featrues
#         pt_pred = self.head(pt_feat)

#         cmb_pred = self.head(cmb_feat)

#         return img_pred, pt_pred, img_feat, pt_feat


class Semi_img_mn(nn.Module):
    def __init__(self, num_classes):
        super(Semi_img_mn, self).__init__()
        self.num_classes = num_classes
        self.branch1 = SingleViewNet(pre_trained=None)
        self.branch2 = SingleViewNet(pre_trained=None)
        self.head1 = nn.Sequential(*[nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, self.num_classes)])
        self.head2 = nn.Sequential(*[nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, self.num_classes)])
    def forward(self, data1,data2, step=1):
        if step == 1:
            img_feat1 = self.branch1(data1,data2)
            img_pred1 = self.head1(img_feat1)
            return img_feat1, img_pred1
        elif step == 2:
            img_feat2 = self.branch2(data1,data2)
            img_pred2 = self.head2(img_feat2)
            return img_feat2, img_pred2
        
class Semi_pt_mn(nn.Module):
    def __init__(self, num_classes=10):
        super(Semi_pt_mn, self).__init__()
        self.num_classes = num_classes
        self.branch1 = DGCNN(self.num_classes)
        self.branch2 = DGCNN(self.num_classes)
        self.head1 = nn.Sequential(*[nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, self.num_classes)])
        self.head2 = nn.Sequential(*[nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, self.num_classes)])
    def forward(self, data, step=1):
        if step == 1:
            pt_feat1 = self.branch1(data)
            pt_pred1 = self.head1(pt_feat1)
            return pt_feat1, pt_pred1
        elif step == 2:
            pt_feat2 = self.branch2(data)
            pt_pred2 = self.head2(pt_feat2)
            return pt_feat2, pt_pred2