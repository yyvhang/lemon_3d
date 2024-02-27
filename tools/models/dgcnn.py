import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, device, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature      # (batch_size, 2*num_dims, num_points, k)

class Transform_Net(nn.Module):
    def __init__(self):
        super(Transform_Net, self).__init__()
        #self.args = args
        self.k = 3

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 3*3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)                   
        x = self.conv2(x)                    
        x = x.max(dim=-1, keepdim=False)[0]  

        x = self.conv3(x)                     
        x = x.max(dim=-1, keepdim=False)[0] 

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)    
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)     

        x = self.transform(x)                
        x = x.view(batch_size, 3, 3)          

        return x


class DGCNN(nn.Module):
    def __init__(self, device, num_class=1, knn_num=20, emb_dim=1024):
        super(DGCNN, self).__init__()
        #self.args = args

        self.device = device
        
        self.seg_num_all = num_class
        self.k = knn_num
        self.transform_net = Transform_Net()
        self.emb_dim = emb_dim

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(self.emb_dim)
        self.bn7 = nn.BatchNorm1d(768)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, self.emb_dim, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(1216, 768, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=0.5)
        

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)
        x0 = get_graph_feature(x, self.device, k=self.k) 
        t = self.transform_net(x0)          
        x = x.transpose(2, 1)                
        x = torch.bmm(x, t)                  
        x = x.transpose(2, 1)                

        x = get_graph_feature(x, self.device, k=self.k) 
        x = self.conv1(x)                    
        x = self.conv2(x)                     
        x1 = x.max(dim=-1, keepdim=False)[0]   

        x = get_graph_feature(x1, self.device, k=self.k)    
        x = self.conv3(x)                    
        x = self.conv4(x)                     
        x2 = x.max(dim=-1, keepdim=False)[0]  

        x = get_graph_feature(x2, self.device, k=self.k)  
        x = self.conv5(x)                     
        x3 = x.max(dim=-1, keepdim=False)[0]  

        x = torch.cat((x1, x2, x3), dim=1)    

        x = self.conv6(x)                      
        Global = x.max(dim=-1, keepdim=True)[0]   

        x = Global.repeat(1, 1, num_points)         

        x = torch.cat((x, x1, x2, x3), dim=1)   

        x = self.conv7(x)                      
        x = self.dp1(x)

        return x


if __name__ == '__main__':

    input = torch.randn(16, 3, 1723).cuda()
    model = DGCNN().cuda()
    out = model(input)
    print(out.size())
