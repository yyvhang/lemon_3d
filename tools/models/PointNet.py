import torch
import torch.nn as nn
import pdb
from tools.utils.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation, PointNetSetAbstractionMsg

class Pts_Encoder_curvature(nn.Module):
    def __init__(self, normal_channel=False):
        super().__init__()

        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0

        self.normal_channel = normal_channel

        self.sa1 = PointNetSetAbstractionMsg(1024, [0.1, 0.2, 0.4], [32, 64, 128], 3+additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(256, [0.4, 0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(npoint=64, radius=0.6, nsample=32, in_channel=512 + 3, mlp=[256, 512, 768], group_all=False)
        
    def forward(self, xyz, curvatures):

        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz
        B = xyz.shape[0]
        l1_xyz, l1_points, fps_idx_1 = self.sa1(l0_xyz, l0_points)
        curvature_1 = curvatures.gather(dim=1, index=fps_idx_1.unsqueeze(-1).expand(B, fps_idx_1.shape[1], 1))

        l2_xyz, l2_points, fps_idx_2 = self.sa2(l1_xyz, l1_points)
        curvature_2 = curvature_1.gather(dim=1, index=fps_idx_2.unsqueeze(-1).expand(B, fps_idx_2.shape[1], 1))

        l3_xyz, l3_points, fps_idx_3 = self.sa3(l2_xyz, l2_points)     # (B, 256, 1024)
        curvature_3 = curvature_2.gather(dim=1, index=fps_idx_3.unsqueeze(-1).expand(B, fps_idx_3.shape[1], 1))

        return [l0_xyz, l1_xyz, l2_xyz, l3_xyz], [l0_points, l1_points, l2_points, l3_points], curvature_3

class Pts_Upsample(nn.Module):
    def __init__(self):
        super().__init__()

        self.fp3 = PointNetFeaturePropagation(in_channel=768+512, mlp=[768, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=256+320, mlp=[256, 256])
        self.fp1 = PointNetFeaturePropagation(in_channel=256+6, mlp=[256, 256])

    def forward(self, pts, feats):

        l2_points = self.fp3(pts[2], pts[3], feats[2], feats[3])                          

        l1_points = self.fp2(pts[1], pts[2], feats[1], l2_points)                          
                        
        l0_points = self.fp1(pts[0], pts[1], torch.cat([pts[0],feats[0]],1), l1_points)     

        return l0_points