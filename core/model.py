import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum
from torchvision import models
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class BaselineModel(nn.Module):
    def __init__(self, out_ch, pretrained):
        super().__init__()
        m = models.resnet50(pretrained=pretrained)
        chilrdens = list(m.children())
        self.pretrained = pretrained
        self.feat_extractor = nn.Sequential(*chilrdens[:8])
        self.avg_pool = list(m.children())[8]
        self.out_proj = nn.Linear(2048, out_ch)
    
    def forward(self, orig, trg):
        trg_feat = self.feat_extractor(trg)
        if self.pretrained:
            trg_feat = trg_feat.detach()
            
        trg_feat = self.avg_pool(trg_feat)
        trg_feat = trg_feat.squeeze(2).squeeze(2)
        out = self.out_proj(trg_feat)
        return out


class BaselineWithOriginalAttentionModel(nn.Module):
    def __init__(self, out_ch, attn_dim):
        super().__init__()
        m = models.resnet50(pretrained=False)
        chilrdens = list(m.children())
        self.feat_extractor = nn.Sequential(*chilrdens[:8])
        self.avg_pool = list(m.children())[8]
        self.cross_attn = CrossAttentionBlock(2048, attn_dim)
        self.out_proj = Out(2048 + attn_dim, out_ch)

        
    def forward(self, orig, trg):
        trg_feat = self.feat_extractor(trg)
        trg_feat = self.avg_pool(trg_feat)
        orig_feat = self.feat_extractor(orig)
                
        attn_out = self.cross_attn(orig_feat, trg_feat)
        trg_feat = trg_feat.squeeze(2).squeeze(2)
        attn_out = torch.cat([trg_feat, attn_out], dim=1)
        
        out = self.out_proj(attn_out)
        return out


class Out(nn.Module):
    def __init__(self, in_dim, out_ch):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_ch)
        self.act = nn.Tanh()
        
    def forward(self, x):
        return self.act(self.fc(x))
    

class CrossAttentionBlock(nn.Module):
    def __init__(self, orig_dim=2048, attn_dim=512):
        super().__init__()

        self.q_proj = nn.Linear(orig_dim, attn_dim, bias=True)
        self.k_proj = nn.Linear(orig_dim, attn_dim, bias=True)
        self.v_proj = nn.Linear(orig_dim, attn_dim, bias=True)
        
        self.scale = attn_dim ** -0.5
        
        self.attn_dim = attn_dim
    
    def forward(self, orig: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
        batch_size, c_trg, _, _ = trg.size()
        _, c_orig, h, w = orig.size()
        
        q = self.q_proj(trg.permute(0, 2, 3, 1))
        k = self.k_proj(orig.permute(0, 2, 3, 1))
        v = self.v_proj(orig.permute(0, 2, 3, 1))
        
        q_flatten = q.view(batch_size, -1, self.attn_dim)
        k_flatten = k.view(1, -1, self.attn_dim)
        
        attn_value = (q_flatten * k_flatten).sum(dim=-1) * self.scale
        
        softmax_attentioned = F.softmax(attn_value.view(batch_size, -1), dim=1).view(
            *attn_value.size()
        )
        
        output_t = softmax_attentioned.unsqueeze(2) * v.view(1, -1, self.attn_dim)
                
        return output_t.sum(dim=1)
