import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum
from torchvision import models
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


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
    

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0., out_dim=None):
        super().__init__()
        
        lin = nn.Linear(hidden_dim, out_dim) if out_dim is not None else nn.Linear(hidden_dim, dim)
        
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            lin,
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


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


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 128, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None, kv_include_self = False):
        b, n, _, h = *x.shape, self.heads
        context = default(context, x)

        if kv_include_self:
            context = torch.cat((x, context), dim = 1)

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

    
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

    
class CrossTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
            )

    def forward(self, tokens):
        trg_cls, context_feat = tokens[:, :1], tokens[:, 1:]

        for attn in self.layers:
            trg_cls = attn(trg_cls, context=context_feat, kv_include_self=True) + trg_cls

        tokens = torch.cat((trg_cls, context_feat), dim = 1)
        return tokens


class ImageEmbedder(nn.Module):
    def __init__(
        self,
        *,
        dim,
        image_size,
        patch_size,
        dropout = 0.
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]

        return self.dropout(x)
    
    
class CnnAttentionMixedModel(nn.Module):
    def __init__(
        self,
        out_ch=5, 
        attn_dim=128, 
        transformer_heads=4, 
        transformer_depth=2,
        dropout=0.1,
        emb_dim=128,
    ):
        super().__init__()
        m = models.resnet50(pretrained=True)
        chilrdens = list(m.children())
        
        self.feat_extractor = nn.Sequential(*chilrdens[:8])
        # self.avg_pool = list(m.children())[8]
        self.avg_pool = nn.AvgPool2d(7)
        self.embedder = nn.Conv2d(in_channels=2048, out_channels=emb_dim, kernel_size=1, stride=1)
        
        self.layers = nn.ModuleList([])
        for _ in range(transformer_depth):
            self.layers.append(
                Transformer(dim=emb_dim, depth=1, heads=transformer_heads, dim_head=attn_dim, mlp_dim=emb_dim, dropout=dropout)
            )
        
        self.out = FeedForward(emb_dim, emb_dim, dropout=0., out_dim=out_ch)
        
    def forward(self, orig, trg):
        b = trg.size(0)
        
        orig_feat = self.feat_extractor(orig)
        trg_feat = self.avg_pool(self.feat_extractor(trg))
            
        orig_feat = self.embedder(orig_feat)
        trg_feat = self.embedder(trg_feat)
        
        orig_feat = rearrange(orig_feat, 'b c h w -> b (h w) c')
        orig_feat = repeat(orig_feat, '() n d -> b n d', b = b)
        trg_feat = rearrange(trg_feat, 'b c h w -> b (h w) c')
        
        tokens = torch.cat([trg_feat, orig_feat], dim=1)
        
        for enc in self.layers:
            tokens = enc(tokens)
                
        out = self.out(tokens[:, 0])
        return out
