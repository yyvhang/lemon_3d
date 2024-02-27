import torch
import torch.nn as nn
import pdb
from einops import rearrange, repeat
from torch.nn import functional as F
from tools.models.dgcnn import DGCNN
from tools.utils.mesh_sampler import get_sample
from tools.models.hrnet.hrnet_cls_net_featmaps import get_cls_net
from tools.models.hrnet.config import update_config as hrnet_update_config
from tools.models.hrnet.config import config as hrnet_config


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class PreNorm_Atten(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, key_value):
        return self.fn(self.norm(x), self.norm(key_value))

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Multi_Branch_Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        self.inner_dim = dim_head *  heads
        self.dim_head = dim_head

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q1 = nn.Linear(dim, self.inner_dim, bias = False)
        self.to_q2 = nn.Linear(dim, self.inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, self.inner_dim*2, bias = False)

    def forward(self, query1, query2, key_value):

        B = query1.size(0)
        q1 = self.to_q1(query1).view(B, -1, self.heads, self.dim_head).permute(0, 2, 1, 3)            #b n (h d)
        q2 = self.to_q2(query2).view(B, -1, self.heads, self.dim_head).permute(0, 2, 1, 3)            #b n (h d)

        kv = self.to_kv(key_value).chunk(2, dim = -1)       
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), kv)

        dots1 = torch.matmul(q1, k.transpose(-1, -2)) * self.scale
        dots2 = torch.matmul(q2, k.transpose(-1, -2)) * self.scale

        attn1 = self.dropout(self.attend(dots1))
        attn2 = self.dropout(self.attend(dots2))

        out1 = torch.matmul(attn1, v)
        out1 = rearrange(out1, 'b h n d -> b n (h d)')

        out2 = torch.matmul(attn2, v)
        out2 = rearrange(out2, 'b h n d -> b n (h d)')
        return out1, out2
    
class Cross_Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        self.inner_dim = dim_head *  heads
        self.dim_head = dim_head

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, self.inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, self.inner_dim*2, bias = False)

    def forward(self, query, key_value):

        B = query.size(0)
        q = self.to_q(query).view(B, -1, self.heads, self.dim_head).permute(0, 2, 1, 3)            #b n (h d)

        kv = self.to_kv(key_value).chunk(2, dim = -1)       
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), kv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.dropout(self.attend(dots))

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm_Atten(dim, Cross_Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, key_value):
        for attn, ff in self.layers:
            x = attn(x, key_value) + x
            x = ff(x) + x
        return x
        
class Enc_I(nn.Module):
    def __init__(self, run_type, hidden_dim):
        super().__init__()

        hrnet_yaml = 'tools/models/hrnet/config/hrnet/cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
        hrnet_update_config(hrnet_config, hrnet_yaml)
        if run_type != 'infer':
            hrnet_checkpoint = 'tools/models/hrnet/config/hrnet/hrnetv2_w40_imagenet_pretrained.pth'
            self.backbone = get_cls_net(hrnet_config, pretrained=hrnet_checkpoint)
        else:
            self.backbone = get_cls_net(hrnet_config, pretrained=None)
        self.hidden_dim = hidden_dim
        self.dim_down = nn.Sequential(
            nn.Conv2d(2048, self.hidden_dim, 1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU()
        )
        self.pooling = nn.AdaptiveAvgPool2d(output_size=(1,1))

    def forward(self, img):
        B = img.size(0)

        out = self.backbone(img)
        out = self.dim_down(out)
        feats = out.view(B, self.hidden_dim, -1).permute(0, 2, 1)
        semantic = self.pooling(out).view(B, -1)
        return feats, semantic

class Intention_Excavation(nn.Module):
    def __init__(self, input_dim, device):
        super().__init__()
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x.transpose(1, 2)
        self.device = device
        self.co_intention = Multi_Branch_Attention(dim = input_dim, heads = 12, dropout = 0.3, dim_head = 64)
        self.T_o = nn.Parameter(torch.zeros(1, 1, input_dim))
        self.T_h = nn.Parameter(torch.zeros(1, 1, input_dim))
        self.cosine = nn.CosineEmbeddingLoss()

    def forward(self ,F_i, F_o, F_h):

        B = F_i.size(0)

        F_to = torch.cat((self.T_o.expand(B,-1,-1), F_o), dim=1)
        F_th = torch.cat((self.T_o.expand(B,-1,-1), F_h), dim=1)

        F_to_, F_th_ = self.co_intention(F_to, F_th, F_i)

        T_o_, T_h_ =  F_to_[:,0,:], F_th_[:,0,:]
        F_o_, F_h_ = F_to_[:,1:,:], F_th_[:,1:,:]

        tar = torch.ones(B).to(self.device)
        varphi = self.cosine(T_o_, T_h_, tar)

        return T_o_, T_h_, F_o_, F_h_, varphi
    
class Curvature_guided_Geometric_Correlation(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x.transpose(1, 2)
        self.hm_curvature_emb = nn.Sequential(
            nn.Conv1d(1, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv1d(128, input_dim, 1),
            nn.BatchNorm1d(input_dim),
            nn.LeakyReLU(negative_slope=0.1),
            SwapAxes(),
        )

        self.obj_curvature_emb = nn.Sequential(
            nn.Conv1d(1, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv1d(128, input_dim, 1),
            nn.BatchNorm1d(input_dim),
            nn.LeakyReLU(negative_slope=0.1),
            SwapAxes(),
        )

        self.f_m1 = Cross_Attention(dim = input_dim, heads = 12, dropout = 0.3, dim_head = 64)
        self.f_m2 = Cross_Attention(dim = input_dim, heads = 12, dropout = 0.3, dim_head = 64)

        self.fusion_hm = nn.Sequential(
            nn.Conv1d(input_dim*2, input_dim, 1),
            nn.BatchNorm1d(input_dim),
            nn.LeakyReLU(negative_slope=0.1),
            SwapAxes(),
        )

        self.fusion_obj = nn.Sequential(
            nn.Conv1d(input_dim*2, input_dim, 1),
            nn.BatchNorm1d(input_dim),
            nn.LeakyReLU(negative_slope=0.1),
            SwapAxes(),
        )

        self.affordance = Transformer(dim = input_dim, depth = 1, heads = 12, mlp_dim = 768, dropout = 0.3, dim_head = 64)
        self.contact = Transformer(dim = input_dim, depth = 1, heads = 12, mlp_dim = 768, dropout = 0.3, dim_head = 64)

    def forward(self, C_h, C_o, F_o_, F_h_, T_o_, T_h_):

        C_o_emb = self.obj_curvature_emb(C_o.mT)               
        C_h_emb = self.hm_curvature_emb(C_h.mT)                   

        C_o_ = self.f_m1(C_o_emb, C_h_emb)
        C_h_ = self.f_m2(C_h_emb, C_o_emb)

        F_co = self.fusion_obj(torch.cat((F_o_, C_o_), dim=2).mT)
        F_ch = self.fusion_hm(torch.cat((F_h_, C_h_), dim=2).mT)

        conditional_aff = torch.cat((T_o_.unsqueeze(dim=1), F_ch), dim=1)
        conditional_contact = torch.cat((T_h_.unsqueeze(dim=1), F_co), dim=1)

        phi_a = self.affordance(F_co, conditional_aff)
        phi_c = self.contact(F_ch, conditional_contact)

        return F_co, F_ch, phi_a, phi_c

class Contact_aware_Spatial_Relation(nn.Module):
    def __init__(self, input_dim, mlp_dim):
        super().__init__()

        self.T_sp = nn.Embedding(3, input_dim)
        self.spatial_pse = nn.Parameter(torch.randn(1, 3+2, input_dim))
        self.f_p = Cross_Attention(dim = input_dim, heads = 12, dropout = 0.3, dim_head = 64)
        self.proj = nn.Sequential(
            nn.Linear(input_dim, input_dim//6),
            nn.BatchNorm1d(input_dim//6),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(input_dim // 6, input_dim//16),
            nn.BatchNorm1d(input_dim//16),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(input_dim // 16, 3),
        )

    def forward(self, F_co, F_ch, phi_c, T_o_, T_h_):
        B = F_co.size(0)
        F_co = F_co.max(dim=1, keepdim=True)[0]
        F_ch = F_ch.max(dim=1, keepdim=True)[0]

        spatial_token = self.T_sp.weight.expand(B,-1,-1)
        spatial_query = torch.cat((spatial_token, F_co, T_o_.unsqueeze(dim=1)), dim=1) + self.spatial_pse.expand(B,-1,-1)

        spatial_kv = torch.cat((phi_c, F_ch, T_h_.unsqueeze(dim=1)), dim=1)
        feats = self.f_p(spatial_query, spatial_kv)
        phi_p = feats[:,0:3,:]         #[B, 3, C]

        return phi_p

class Decoder(nn.Module):
    def __init__(self, feat_dim, device):
        super().__init__()
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x.transpose(1, 2)
        
        self.device = device
        self.aff_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim//6),
            SwapAxes(),
            nn.BatchNorm1d(feat_dim//6),
            SwapAxes(),
            nn.ReLU(),
            nn.Linear(feat_dim//6, 1)
        )

        self.contact_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim//6),
            SwapAxes(),
            nn.BatchNorm1d(feat_dim//6),
            SwapAxes(),
            nn.ReLU(),
            nn.Linear(feat_dim//6, 1)
        )

        self.spatial_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim//6),
            SwapAxes(),
            nn.BatchNorm1d(feat_dim//6),
            SwapAxes(),
            nn.ReLU(),
            nn.Linear(feat_dim//6, 1)
        )

        self.content = nn.Sequential(
            nn.Linear(feat_dim, feat_dim//6),
            nn.BatchNorm1d(feat_dim//6),
            nn.ReLU(),
            nn.Linear(feat_dim//6, 17)           
        )


        self.contact_up_fine = nn.Linear(1723, 6890)
        self.sigmoid = nn.Sigmoid()

    def forward(self, phi_a, phi_c, phi_p, semantic_feats):

        B = phi_a.size(0)
        affordance = self.aff_head(phi_a)                                 
        affordance = self.sigmoid(affordance)

        contact_coarse = self.contact_head(phi_c)
        contact_fine = self.contact_up_fine(contact_coarse.mT)            

        spatial = self.spatial_head(phi_p).squeeze(dim=-1)

        semantic = self.content(semantic_feats)

        return [self.sigmoid(contact_coarse), self.sigmoid(contact_fine.mT)], affordance, spatial, semantic

class LEMON(nn.Module):
    def __init__(self, feat_dim, run_type, device):
        super().__init__()
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x.transpose(1, 2)

        self.emb_dim = 1024
        self.device = device
        self.vertex_sampler = get_sample(device=self.device)

        self.img_encoder = Enc_I(run_type, feat_dim)
        self.obj_encoder = DGCNN(device=self.device, emb_dim=self.emb_dim)
        self.hm_encoder = DGCNN(device=self.device, emb_dim=self.emb_dim)

        self.Intention = Intention_Excavation(feat_dim, device)
        self.Geometry_Correlation = Curvature_guided_Geometric_Correlation(feat_dim)
        self.Spatial = Contact_aware_Spatial_Relation(feat_dim, feat_dim)

        self.decoder = Decoder(feat_dim, device=self.device)

    def forward(self, I, O, H, C_h, C_o, meta_masks=None):

        B = I.size(0)
        H = self.vertex_sampler.downsample(H, n1=0, n2=1)                     
        if meta_masks != None:
            constant_tensor = torch.ones_like(H).to(self.device)*0.01
            H = H*meta_masks + constant_tensor*(1-meta_masks)
        F_i, semantic_feats = self.img_encoder(I)                     
        F_o = self.obj_encoder(O)                                     
        F_h = self.hm_encoder(H.mT)                                     

        T_o_, T_h_, F_o_, F_h_, varphi = self.Intention(F_i, F_o.mT, F_h.mT)
        F_co, F_ch, phi_a, phi_c = self.Geometry_Correlation(C_h, C_o, F_o_, F_h_, T_o_, T_h_)
        phi_p = self.Spatial(F_co, F_ch, phi_c, T_o_, T_h_)

        contact, affordance, spatial, semantic = self.decoder(phi_a, phi_c, phi_p, semantic_feats)

        return contact, affordance, spatial, semantic, varphi

if __name__=='__main__':
    pass
