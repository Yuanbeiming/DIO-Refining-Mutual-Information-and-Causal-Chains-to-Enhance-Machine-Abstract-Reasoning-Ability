import torch
from torch import einsum
import torch.nn as nn
from einops import rearrange, repeat
import math
import numpy as np
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import torch.distributed as distri
from pathlib import Path
import atexit
import signal
import sys
import os
from typing import Optional, Union, Dict, Any
import datetime


def conv1x1(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)


def get_attn_pad_mask(seq_q, seq_k, mask_value = 0):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(mask_value).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]

def get_rpm_attn_pad_mask(seq_q, seq_k, mask_value = 13):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(mask_value).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
    pad_attn_mask += torch.cat((pad_attn_mask[:,:,1:], pad_attn_mask[:,:,0:1]), dim = 2)
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(in_channel, out_channel, stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        if downsample is None:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(conv3x3(in_channel, out_channel, stride),nn.BatchNorm2d(out_channel))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.downsample(x) + self.bn2(self.conv2(out)))

        return out


class ResBlock1x1(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, downsample = None):
        super(ResBlock1x1, self).__init__()
        self.conv1 = conv1x1(in_channel, out_channel, stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x1(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        
        if downsample is None:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(conv3x3(in_channel, out_channel, stride),nn.BatchNorm2d(out_channel))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.downsample(x) + self.bn2(self.conv2(out)))

        return out



class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)






class take_cls(nn.Module):
    def __init__(self, num_cls = 1, keepdim = False):
        super(take_cls, self).__init__()
        self.keepdim = keepdim
        self.num_cls = num_cls
        if keepdim == False:
            
            assert num_cls == 1

    def forward(self, x):
        if self.keepdim == False:
            return x[:,0]
        
        else:
            return x[:,0:self.num_cls]






class Mean(nn.Module):
    def __init__(self, dim, keepdim = False):
        super(Mean, self).__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return x.mean(dim = self.dim, keepdim = self.keepdim)

class Bottleneck(nn.Module):
    def __init__(self,in_places,places, stride=1, downsampling=False, expansion = 4):
        super(Bottleneck,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places*self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        


        if self.downsampling:
            residual = self.downsample(x.contiguous())
            


        out += residual
        out = self.relu(out)
        return out
    
    
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x,**kwargs):
        return self.fn(self.norm(x), **kwargs)
    
    
class Mask_PreNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        return self.norm(x)
    

class Mask_PreNorm_shell(nn.Module):
    def __init__(self, dim, attn_fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn_fn = attn_fn

    def forward(self, x, **kwargs):

        
        return self.attn_fn(self.norm(x), **kwargs)

       
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        # hidden_dim = mlp_dim
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x, *args):
        return self.net(x)
    


# Attention
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 32, dropout = 0.):
        #dim_head: dim of signle head
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
 
 
        self.heads = heads
        self.scale = dim_head ** -0.5
 
 
        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
 
 
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
 
 
    def forward(self, x, **kwargs):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
    
class Cross_Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 32, dropout = 0.):
        #dim_head: dim of signle head
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
 
 
        self.heads = heads
        self.scale = dim_head ** -0.5
 
 
        self.attend = nn.Softmax(dim = -1)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        
        self.kv_norm = nn.LayerNorm(dim)
        
        self.q_norm = nn.LayerNorm(dim)
 
 
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
 
 
    def forward(self, q, kv, **kwargs):
        h =  self.heads
        
        q = self.q_norm(q)
        
        kv = self.kv_norm(kv)
        qkv = [q, *self.to_kv(kv).chunk(2, dim = -1)]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
 

        
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.dim = dim

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.dim) # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.
        
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context, attn

    

    
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

    
class value_Attention(nn.Module):
    def __init__(self, dim, out_dim, heads = 8, dim_head = 32, dropout = 0.):
        #dim_head: dim of signle head
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        
        # print(project_out)
        
        assert out_dim % heads == 0
        
        
        self.dim = dim
        
        self.out_dim = out_dim
 
 
        self.heads = heads
        self.scale = dim_head ** -0.5
 
 
        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(inner_dim, inner_dim * 2 + out_dim, bias = False)
 
 
        self.to_out = nn.Identity()
 
 
    def forward(self, x, **kwargs):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).split([self.dim, self.dim, self.out_dim], dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        # print(out.shape)
        return self.to_out(out)    

    
class Mask_Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 32, dropout = 0.):
        #dim_head: dim of signle head
        super(Mask_Attention, self).__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
 
 
        self.heads = heads
        self.scale = dim_head ** -0.5
 
 
        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
 
 
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
 
 
    def forward(self, x, attn_mask):

 
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.heads, 1, 1)
        
        
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        dots.masked_fill_(attn_mask, -1e9)
        
        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)    

class pose_matrix(nn.Module):
    def __init__(self, words, dim, heads, dim_head):
        #dim_head: dim of signle head
        super().__init__()

        
        
        
        self.weight = nn.Parameter(.001*torch.randn(1,
                                            heads,   #6x6x32  
                                            words,       #16
                                            dim,       #256
                                            dim_head) )     #32
   
    def forward(self, x, **kwargs):
       
        
        x = self.weight@x #b, n*h, n, d, 1
        
    
        
        return x#b, h*n, d*a

    
class Routing(nn.Module):
    def __init__(self, iterations, words, dim, heads = 8, dim_head = 32, alpha = 1, matrix_norm = False):
        #dim_head: dim of signle head
        super().__init__()

        self.iterations = iterations
        
        self.words = words

        
        inner_dim = int(dim*alpha)
        project_out = inner_dim == dim
 
 
        self.heads = int(heads*alpha)
        
        self.dim_head = dim_head
        
        
        if matrix_norm:
            self.W =  nn.utils.spectral_norm(
                
                pose_matrix(words = words, dim = inner_dim, heads = heads, dim_head = dim_head)
                
                )

        else:
            self.W = nn.Parameter(.001*torch.randn(1,
                                                heads,   #6x6x32  
                                                words,       #16
                                                inner_dim,       #256
                                                dim_head) )     #32     #32

        self.matrix_norm = matrix_norm                    
                                                
        self.bias = 0
        
        self.inner = nn.Identity() if project_out else nn.Linear(dim, inner_dim)
        
        self.outer = nn.Identity() if project_out else nn.Linear(inner_dim, dim)
 
 
 
    def routing(self, u_hat):
        b = torch.zeros_like( u_hat )#同尺寸可以内积相乘 torch.Size([256, 1152, 10, 16])
        u_hat_routing = u_hat.detach()  #前两次禁止回传
        for i in range(self.iterations):#3次迭代
            c = F.softmax(b, dim=2)   #在第三维度上进行softmax,10类连接强度，初始值1/10 torch.Size([256, 1152, 10, 16]
            if i==(self.iterations-1):#最后一次迭代保存梯度
                s = (c*u_hat).sum(1, keepdim=True)  #tor, input_size = 201ch.Size([256, 1, 10, 16]
            else:
                s = (c*u_hat_routing).sum(1, keepdim=True)   #torch.Size([256, 1, 10, 16]
            v = self.squash(s)  #torch.Size([256, 1, 10, 16]，投票结果
            if i < self.iterations - 1: #并未最后一次迭代                             v                      u_hat_routing                u_hat_routing与v的内积
                b = (b + (u_hat_routing*v).sum(3, keepdim=True))#(torch.Size([256, 1, 10, 16]*torch.Size([256, 1152, 10, 16])→ torch.Size([256, 1152, 10, 1]))+torch.Size([256, 1152, 10, 16])
        return v  #torch.Size([256, 1, 10, 16])  
            
    def squash(self, s):
        s_norm = s.norm(dim=-1, keepdim=True)
        v = s_norm / (1. + s_norm**2) * s
        return v    
    def forward(self, x, **kwargs):
        b, n, d, h = *x.shape, self.heads
        
        assert n == 1
        
        x = self.inner(x)

        x = rearrange(x, 'b n (h d) -> b (n h) () d ()', h = self.heads, d = self.dim_head)
        
        
        x = self.W(x) if self.matrix_norm else self.W@x #b, n*h, n, d, 1
        
        x = x.squeeze(-1) + self.bias
        
        x = self.routing(x)
        
        x = x.squeeze(1)
        
        return self.outer(x)#b, h*n, d*a
    
class Routing_norm(Routing):
    def __init__(self, iterations, words, dim, heads = 8, dim_head = 32, alpha = 1):
        #dim_head: dim of signle head
        super().__init__(iterations, words, dim, heads, dim_head, alpha)

        self.layernorm = nn.LayerNorm(dim)

            
    def squash(self, s):
        v = self.layernorm(s)
        return v    
    
        
class Routing_Transformer_layer(nn.Module):
    def __init__(self, iterations, words, dim, depth, heads, dim_head, mlp_dim, alpha, dropout = 0.):
        super().__init__()
        #self.param = nn.Parameter(1e-8*torch.ones(depth))  
        self.param = torch.ones(depth)#32
        self.layers = nn.ModuleList([])
        
        self.cls_tokens = nn.Parameter(torch.randn(1,1,dim))
        
        for _ in range(depth):#L
            self.layers.append(nn.ModuleList([

                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)),
                
                PreNorm(dim, Routing(iterations = iterations, words = words, dim = dim, heads = heads, dim_head = dim_head, alpha = alpha)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)),
            ]))
            
        self.Transformer = Transformer(dim, 1, heads, dim_head, mlp_dim, dropout = dropout)
    def forward(self, x):
        
        b, n, d = x.shape
        
        cls_tokens = self.cls_tokens.expand(b,-1,-1)
        
        x = torch.cat((cls_tokens, x), dim = 1)
        
        for index, (attn, ff_0, routing, ff_1)  in enumerate(self.layers):
           
            x = attn(x, name = 'yuanbeiming', name_didi = 'chendiancheng') + x
            x = ff_0(x) + x
            
            x_cls, x_tokens = x.split([1,n], dim = 1)
            
            x_tokens = routing(x_cls) + x_tokens
            
            x = torch.cat((x_cls, x_tokens), dim = 1)
            
            x = ff_1(x) + x
        
        x = self.Transformer(x[:,1:])

        return x
    
class Routing_mask_Transformer_layer(nn.Module):
    def __init__(self, iterations, words, dim, depth, heads, dim_head, mlp_dim, alpha, dropout = 0.):
        super().__init__()
        #self.param = nn.Parameter(1e-8*torch.ones(depth))  
        self.param = torch.ones(depth)#32
        self.layers = nn.ModuleList([])
        
        self.cls_tokens = nn.Parameter(torch.randn(1,1,dim))
        
        for _ in range(depth):#L
            self.layers.append(nn.ModuleList([

                # Mask_PreNorm(dim),
                PreNorm(dim, Mask_Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)),
                
                PreNorm(dim, Routing(iterations = iterations, words = words, dim = dim, heads = heads, dim_head = dim_head, alpha = alpha)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)),
            ]))
            
        self.Transformer = Transformer(dim, 1, heads, dim_head, mlp_dim, dropout = dropout)
    def forward(self, x):
        
        b, n, d = x.shape
        
        attn_mask = torch.ones(b, n+1, n+1, requires_grad= False).to(x.device)
        
        attn_mask[:,:,:1] = 0
        
        attn_mask = attn_mask.eq(0)
        

        
        cls_tokens = self.cls_tokens.expand(b,-1,-1)
        
        x = torch.cat((cls_tokens, x), dim = 1)
        
        for index, (mask_attn, ff_0, routing, ff_1)  in enumerate(self.layers):
            
        
            x = mask_attn(x, attn_mask = attn_mask) + x
            x = ff_0(x) + x
            
            x_cls, x_tokens = x.split([1,n], dim = 1)
            
            x_tokens = routing(x_cls) + x_tokens
            
            x = torch.cat((x_cls, x_tokens), dim = 1)
            
            x = ff_1(x) + x
        
        x = self.Transformer(x[:,1:])

        return x
#%%
class Cross_Transformer_layer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):#L
            self.layers.append(nn.ModuleList([
                
                Cross_Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)),
                
                
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, q, kv):
        
        # print(q.shape)
        for c_attn, ff, attn, ff_1 in self.layers:
            
            q = c_attn(q, kv) + q
            q = ff(q) + q
            
            kv = attn(kv, name = 'yuanbeiming', name_didi = 'chendiancheng') + kv
            kv = ff_1(kv) + kv
        return q
    
class Cross_Transformer(nn.Module):
    def __init__(self, words, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, words, dim))
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Cross_Transformer_layer(dim, depth, heads, dim_head, mlp_dim, dropout = 0.)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, kv):
        b,n,d = kv.shape
        kv += self.pos_embedding[:, :]
        # dropout
        kv = self.dropout(kv)

        x = self.transformer(q, kv)

        return x
#%%
class Routing_Transformer(nn.Module):
    def __init__(self, iterations, words, dim, depth, heads, dim_head, mlp_dim, alpha, dropout = 0.):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, words, dim))
        self.transformer = Routing_Transformer_layer(iterations, words, dim, depth, heads, dim_head, mlp_dim, alpha, dropout = dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        b,n,d = x.shape
        x += self.pos_embedding[:, :]
        # dropout
        x = self.dropout(x)

        x = self.transformer(x)

        return x
    
    
class Routing_mask_Transformer(nn.Module):
    def __init__(self, iterations, words, dim, depth, heads, dim_head, mlp_dim, alpha, dropout = 0.):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, words, dim))
        self.transformer = Routing_mask_Transformer_layer(iterations, words, dim, depth, heads, dim_head, mlp_dim, alpha, dropout = dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        b,n,d = x.shape
        x += self.pos_embedding[:, :]
        # dropout
        x = self.dropout(x)

        x = self.transformer(x)

        return x
    
class value_Transformer(nn.Module):
    def __init__(self, dim, out_dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        
        assert depth == 1
        self.layers = nn.ModuleList([])
        
        self.to_value_dim = nn.Linear(dim, out_dim)
        
        for _ in range(depth):#L
            self.layers.append(nn.ModuleList([
                PreNorm(dim, value_Attention(dim, out_dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(out_dim, FeedForward(out_dim, out_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x, name = 'yuanbeiming', name_didi = 'chendiancheng') + self.to_value_dim(x)
            x = ff(x) + x
        return x


#%%   
# 基于PreNorm、Attention和FFN搭建Transformer
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):#L
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x, name = 'yuanbeiming', name_didi = 'chendiancheng') + x
            x = ff(x) + x
        return x

  
    
class Mask_Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super(Mask_Transformer,self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):#L
            self.layers.append(nn.ModuleList([
                 Mask_PreNorm(dim),
                 Mask_Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                 Mask_PreNorm(dim),
                 FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x, attn_mask):
        for p_1, attn, p_2, ff in self.layers:
            
            x = p_1(x)
            x = attn(x, attn_mask) + x
            
            x = p_2(x)
            x = ff(x) + x
        return x



    
class graph_transformer(nn.Module):
    def __init__(self, words, dim, depth, heads, dim_head, mlp_dim, num_cls = 1, dropout = 0., PositionalEncoding = True):
        super(graph_transformer,self).__init__()
        
        
        self.PositionalEncoding = PositionalEncoding
        if  self.PositionalEncoding:
            self.pos_embedding = nn.Parameter(torch.randn(1, words + num_cls, dim))
            
        
        self.num_cls = num_cls
        if num_cls != 0:
            self.cls_token = nn.Parameter(torch.randn(1, num_cls, dim))
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout = 0.)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        b,n,d = x.shape
        
        if self.num_cls != 0:
            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
            x = torch.cat((cls_tokens, x), dim=1)
            
        if  self.PositionalEncoding:
            x += self.pos_embedding[:, :]
        # dropout
        x = self.dropout(x)

        x = self.transformer(x)

        return x

class graph_mask_transformer(nn.Module):
    def __init__(self, words, dim, depth, heads, dim_head, mlp_dim, num_cls = 1, dropout = 0.):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, words + num_cls, dim))
        self.num_cls = num_cls
        if num_cls != 0:
            self.cls_token = nn.Parameter(torch.randn(1, num_cls, dim))
        self.transformer = Mask_Transformer(dim, depth, heads, dim_head, mlp_dim, dropout = 0.)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):#b,n
        b,n,d = x.shape
        
        attn_mask = torch.ones(b, n + self.num_cls, n + self.num_cls, requires_grad= False).to(x.device)
        
        attn_mask[:,:,:self.num_cls] = 0
        
        if self.num_cls != 0:
        
            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
    
            x = torch.cat((cls_tokens, x), dim=1)
        
        
        x += self.pos_embedding[:, :].to(x.device)
        # dropout
        x = self.dropout(x)

        x = self.transformer(x, attn_mask.eq(0))

        return x  

class txt_transformer(nn.Module):
    def __init__(self, words,  dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super(txt_transformer,self).__init__()
        assert dim%2 == 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.pos_embedding = torch.zeros(words + 1, dim)
        
        
        position = torch.arange(0, words + 1, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))

        self.pos_embedding[:, 0::2] = torch.sin(position * div_term)
        self.pos_embedding[:, 1::2] = torch.cos(position * div_term)

        self.pos_embedding = self.pos_embedding.unsqueeze(0)
        
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout = 0.)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        b,n,d = x.shape
        
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        
        
        x += self.pos_embedding[:, :].to(x.device)
        # dropout
        x = self.dropout(x)

        x = self.transformer(x)

        return x    


class txt_mask_transformer(nn.Module):
    def __init__(self, dict_size, words,  dim, depth, 
                 heads, dim_head, mlp_dim, dropout = 0., is_pgm = False, num_cls = 0):
        super(txt_mask_transformer,self).__init__()
        assert dim%2 == 0
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.pos_embedding = torch.zeros(words + num_cls, dim)
        self.Enbedding = nn.Embedding(dict_size, dim)
        
        self.is_pgm = is_pgm
        self.num_cls = num_cls
        
        position = torch.arange(0, words + num_cls, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))

        self.pos_embedding[:, 0::2] = torch.sin(position * div_term)
        self.pos_embedding[:, 1::2] = torch.cos(position * div_term)

        self.pos_embedding = self.pos_embedding.unsqueeze(0)
        
        if self.num_cls != 0:
            self.cls_token = nn.Parameter(torch.randn(1, num_cls, dim))
        self.transformer = Mask_Transformer(dim, depth, heads, dim_head, mlp_dim, dropout = 0.)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, seq):#b,n
        x = self.Enbedding(seq)
        b,n,d = x.shape
        
        if self.num_cls != 0:
            seq = torch.cat((torch.ones(b,self.num_cls).to(seq.device), seq), dim = 1) 
        
        if self.is_pgm:
            attn_mask = get_rpm_attn_pad_mask(seq,seq,13)
        else:
            attn_mask = get_rpm_attn_pad_mask(seq,seq,0)
        
        if self.num_cls != 0:
            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
            
                
                
            x = torch.cat((cls_tokens, x), dim=1)
        
        
        x += self.pos_embedding[:, :].to(x.device)
        # dropout
        x = self.dropout(x)
        


        x = self.transformer(x, attn_mask)

        return x    


  
#%%   
def pair(t):
    return t if isinstance(t, tuple) else (t, t)       
        
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size,  dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        #dim: lengh of token ，depth： depth of tranformer, dim_head: dim of signle head, mlp_dim: dim of mlp of transfomer
        super().__init__()
        image_height, image_width = image_size if isinstance(image_size, tuple) or  isinstance(image_size, list) else pair(image_size)
        patch_height, patch_width = pair(patch_size)
     
 
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        # patch数量
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        # patch维度
        patch_dim = channels * patch_height * patch_width
        
        # 定义块嵌入
        self.name = 'ViT'
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )
        # 定义位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        # 定义类别向量

        self.dropout = nn.Dropout(emb_dropout)
 
 
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
 
 
        self.to_latent = nn.Identity()
        # 定义MLP

    # ViT前向流程
    def forward(self, img):
        # 块嵌入
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # 追加位置编码
        # print(x)
        x += self.pos_embedding[:, :n]
        # dropout
        x = self.dropout(x)
        # 输入到transformer
        x = self.transformer(x)
        # x_ = x.mean(dim = 1, keepdim = True) if self.pool == 'mean' else x[:,1:(n + 1)]
        # MLP
        return x   


        

class Routing_ViT(nn.Module):
    def __init__(self, *, iterations, image_size, patch_size,  dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, alpha = 0.5, dropout = 0., emb_dropout = 0.):
        #dim: lengh of token ，depth： depth of tranformer, dim_head: dim of signle head, mlp_dim: dim of mlp of transfomer
        super().__init__()
        image_height, image_width = image_size if isinstance(image_size, tuple) or  isinstance(image_size, list) else pair(image_size)
        patch_height, patch_width = pair(patch_size)
     
 
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        # patch数量
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        # patch维度
        patch_dim = channels * patch_height * patch_width
        
        # 定义块嵌入
        self.name = 'ViT'
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )
        # 定义位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        # 定义类别向量

        self.dropout = nn.Dropout(emb_dropout)
 
 
        # self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.transformer = Routing_Transformer_layer(iterations, num_patches, dim, depth, heads, dim_head, mlp_dim, alpha, dropout = dropout)
 
 
        self.to_latent = nn.Identity()
        # 定义MLP

    # ViT前向流程
    def forward(self, img):
        # 块嵌入
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # 追加位置编码
        # print(x)
        x += self.pos_embedding[:, :n]
        # dropout
        x = self.dropout(x)
        # 输入到transformer
        x = self.transformer(x)
        # x_ = x.mean(dim = 1, keepdim = True) if self.pool == 'mean' else x[:,1:(n + 1)]
        # MLP
        return x   
    
    
class Routing_mask_ViT(Routing_ViT):
    def __init__(self, *, iterations, image_size, patch_size,  dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, alpha = 0.5, dropout = 0., emb_dropout = 0.):
        #dim: lengh of token ，depth： depth of tranformer, dim_head: dim of signle head, mlp_dim: dim of mlp of transfomer
        super().__init__(iterations = iterations,
                                    image_size = image_size,
                                    patch_size = patch_size,
                                    dim = dim, 
                                    depth = depth,
                                    heads = heads,
                                    mlp_dim = mlp_dim,
                                    channels = channels,
                                    dim_head = dim_head,
                                    alpha = alpha,
                                    dropout = dropout,
                                    emb_dropout = dropout)
        
        image_height, image_width = image_size if isinstance(image_size, tuple) or  isinstance(image_size, list) else pair(image_size)
        patch_height, patch_width = pair(patch_size)
     
 
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        # patch数量
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        # patch维度
        patch_dim = channels * patch_height * patch_width
        
        
        self.transformer = Routing_mask_Transformer_layer(iterations, num_patches, dim, depth, heads, dim_head, mlp_dim, alpha, dropout = dropout)

        # 定义MLP

    # ViT前向流程
    def forward(self, img):
        # 块嵌入
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # 追加位置编码
        # print(x)
        x += self.pos_embedding[:, :n]
        # dropout
        x = self.dropout(x)
        # 输入到transformer
        x = self.transformer(x)
        # x_ = x.mean(dim = 1, keepdim = True) if self.pool == 'mean' else x[:,1:(n + 1)]
        # MLP
        return x   


class ViT_reverse(nn.Module):
    def __init__(self, *, image_size, patch_size,  dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        #dim: lengh of token ，depth： depth of tranformer, dim_head: dim of signle head, mlp_dim: dim of mlp of transfomer
        super().__init__()
        image_height, image_width = image_size if isinstance(image_size, tuple) or  isinstance(image_size, list) else pair(image_size)
        patch_height, patch_width = pair(patch_size)
     
 
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        # patch数量
        
        h, w = (image_height // patch_height), (image_height // patch_height)
        num_patches = h* w
        # patch维度
        patch_dim = channels * patch_height * patch_width
        
        # 定义块嵌入
        self.name = 'ViT_reverse'
        self.to_patch_embedding = Rearrange('b (m n) (c p1 p2) -> b c (m p1) (n p2)', p1 = patch_height, p2 = patch_width, c = channels, m = image_height // patch_height, n = image_width // patch_width)
            
        # )
        # 定义位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        # 定义类别向量

        self.dropout = nn.Dropout(emb_dropout)
 
        self.input_transformer = value_Transformer(dim, patch_dim, 1, heads, dim_head, patch_dim, dropout)
        

        self.transformer = Transformer(patch_dim, depth - 1, heads, dim_head, patch_dim, dropout) if depth > 1 else nn.Identity()
 
 
        self.to_latent = nn.Identity()
        # 定义MLP

    # ViT前向流程
    def forward(self, img):
        # 块嵌入
        
        x  = img

        b, n, _ = x.shape
        


        # 追加位置编码
        # print(x)
        x = x + self.pos_embedding[:, :n]
        # dropout
        x = self.dropout(x)
        # 输入到transformer
        
        x = self.input_transformer(x)
        x = self.transformer(x)
        
        
        x = self.to_patch_embedding(x)
        # x_ = x.mean(dim = 1, keepdim = True) if self.pool == 'mean' else x[:,1:(n + 1)]
        # MLP
        return x  
    
class ViT_reverse_aux_linear(nn.Module):
    def __init__(self, *, image_size, patch_size,  dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        #dim: lengh of token ，depth： depth of tranformer, dim_head: dim of signle head, mlp_dim: dim of mlp of transfomer
        super().__init__()
        image_height, image_width = image_size if isinstance(image_size, tuple) or  isinstance(image_size, list) else pair(image_size)
        patch_height, patch_width = pair(patch_size)
     
 
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        # patch数量
        
        h, w = (image_height // patch_height), (image_height // patch_height)
        num_patches = h* w
        # patch维度
        patch_dim = channels * patch_height * patch_width
        
        m = image_height // patch_height
        
        n= image_width // patch_width
        
        # 定义块嵌入
        self.name = 'ViT_reverse'
        self.to_patch_embedding = Rearrange('b (m n) (c p1 p2) -> b c (m p1) (n p2)', p1 = patch_height, p2 = patch_width, c = channels, m = image_height // patch_height, n = image_width // patch_width)
            
        # )
        # 定义位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        # 定义类别向量

        self.dropout = nn.Dropout(emb_dropout)
 
        self.input_transformer = value_Transformer(dim, patch_dim, 1, heads, dim_head, patch_dim, dropout)
        

        self.transformer = Transformer(patch_dim, depth - 1, heads, dim_head, patch_dim, dropout) if depth > 1 else nn.Identity()
 
    
        self.lin = nn.Sequential(Rearrange('b s d -> b (s d)'),
                                 nn.GELU(),
                                 nn.Linear(patch_dim*m*n, patch_dim*m*n),
                                 Rearrange('b (s d) -> b s d', s=(m*n))
            
            )
 
        self.to_latent = nn.Identity()
        # 定义MLP

    # ViT前向流程
    def forward(self, img):
        # 块嵌入
        
        x  = img

        b, n, _ = x.shape
        


        # 追加位置编码
        # print(x)
        x = x + self.pos_embedding[:, :n]
        
        # dropout
        x = self.dropout(x)
        # 输入到transformer
        
        x = self.input_transformer(x)
        x = self.transformer(x)
        
        x = self.lin(x)
        
        x = self.to_patch_embedding(x)
        # x_ = x.mean(dim = 1, keepdim = True) if self.pool == 'mean' else x[:,1:(n + 1)]
        # MLP
        return x  
    

    
    
    
class ViT_reverse_with_cls(nn.Module):
    def __init__(self, *, image_size, patch_size,  dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        #dim: lengh of token ，depth： depth of tranformer, dim_head: dim of signle head, mlp_dim: dim of mlp of transfomer
        super().__init__()
        image_height, image_width = image_size if isinstance(image_size, tuple) or  isinstance(image_size, list) else pair(image_size)
        patch_height, patch_width = pair(patch_size)
     
 
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        # patch数量
        
        h, w = (image_height // patch_height), (image_height // patch_height)
        num_patches = h* w
        
        
        self.num_cls = num_cls = num_patches

        # patch维度
        patch_dim = channels * patch_height * patch_width
        
        # 定义块嵌入
        self.name = 'ViT_reverse'
        self.num_cls = num_cls
        self.to_patch_embedding = Rearrange('b (m n) (c p1 p2) -> b c (m p1) (n p2)', p1 = patch_height, p2 = patch_width, c = channels, m = image_height // patch_height, n = image_width // patch_width)
            
        # )
        # 定义位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + num_cls, dim))
        # 定义类别向量
        

        self.cls_token = nn.Parameter(torch.randn(1, num_cls, dim))
        

        self.dropout = nn.Dropout(emb_dropout)
 
        self.input_transformer = value_Transformer(dim, patch_dim, 1, heads, dim_head, patch_dim, dropout)
        
        # if depth >= 2:
            
        #     self.transformer = Transformer(patch_dim, depth - 1, heads, int(patch_dim/heads), patch_dim, dropout)
            
        # else:
            
        #     self.transformer = nn.Identity()
            
            
        self.transformer = Transformer(patch_dim, depth - 1, heads, dim_head, patch_dim, dropout) if depth > 1 else nn.Identity()
 
 
        self.to_latent = nn.Identity()
        # 定义MLP

    # ViT前向流程
    def forward(self, img):
        # 块嵌入
        
        x  = img

        b, n, _ = x.shape
        
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
            
        x = torch.cat((cls_tokens, x), dim=1)

        # 追加位置编码
        # print(x)
        x += self.pos_embedding[:, :n + self.num_cls]
        
        if self.training and np.random.rand() < 0.33: 
            x = self.add_noise_to_patch(x, max_drop=int(x.shape[1]/4) )
        # dropout
        x = self.dropout(x)
        # 输入到transformer
        
        x = self.input_transformer(x)
        x = self.transformer(x)[:,:self.num_cls]
        
        
        x = self.to_patch_embedding(x)
        # x_ = x.mean(dim = 1, keepdim = True) if self.pool == 'mean' else x[:,1:(n + 1)]
        # MLP
        return x 
    
    def add_noise_to_patch(self, x, min_noise=1, max_noise=8, noise_scale=0.1, zero_padding_p = 0.3):
        b, s, d = x.shape
        device = x.device
        
        # 每个batch随机决定加噪个数
        count = torch.randint(min_noise, max_noise + 1, (b,), device=device)
        
        zero_padding = (torch.rand(b, device=device) > zero_padding_p).float()
        
        count = count * zero_padding
        
        # 在后16个位置(16-31)中随机选
        rand = torch.rand(b, self.num_cls, device=device)
        ranks = rand.argsort(dim=1).argsort(dim=1)  # 排名
        
        # 局部mask (b, 16)
        mask_local = ranks < count.unsqueeze(1)
        
        # 拼成全局mask (b, 32)
        mask = torch.zeros(b, self.num_cls + self.num_cls, dtype=torch.bool, device=device)
        mask[:, self.num_cls:] = mask_local  # 后16位置
        
        # 加噪
        noise = torch.randn_like(x) * noise_scale
        return torch.where(mask.unsqueeze(-1).expand(-1, -1, d), x + noise, x), count


class Mixed_gussan(nn.Module):
    def __init__(self, words, dim, depth_block, depth,  heads,
                              dim_head, mlp_dim, dropout=0.1):
        super().__init__()

        assert depth >= 2, 'depth error'

        self.low_dim = dim

        self.encoder = nn.ModuleList([])

        self.num_cls = [0] + (depth - 1) *[1]

        for i in range(depth):

            self.encoder.append(graph_transformer(words=words + self.num_cls[i], dim=self.low_dim, depth=depth_block, heads=heads,
                              dim_head=dim_head, mlp_dim=mlp_dim, num_cls = 2, dropout=dropout),)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std



    def forward(self, x):

        b, _, d = x.shape

        sample = torch.empty(b, 0, d).to(x.device)
        for layer in self.encoder:

            mu_logvar = layer( torch.cat([x, sample]), dim = 1)[:,:2]
            sample = self.reparameterize(mu_logvar[:, 0:1], mu_logvar[:, 1:2])

        return sample, mu_logvar


        
    
class Bottleneck_judge(nn.Module):
    def __init__(self, in_places, hidden_places, out_places = 1,  dropout = 0.1, last_dropout = 0.5):
        super(Bottleneck_judge,self).__init__()



        self.bottleneck = nn.Sequential(
            nn.Linear(in_places, hidden_places),
            nn.GELU(),
            nn.BatchNorm1d(hidden_places),
            nn.Linear(hidden_places, hidden_places),
            nn.GELU(),
            nn.BatchNorm1d(hidden_places),
            nn.Linear(hidden_places, out_places)
        )

        if in_places != out_places:
            self.downsample = nn.Sequential(
                nn.Linear(in_places, out_places)
            )
            
        else:
            self.downsample = nn.Identity()
            

    def forward(self, x):

        out = self.bottleneck(x)
        
        residual = self.downsample(x)
        
        out += residual
        
        return out
    
class Bottleneck_judge_II(nn.Module):
    def __init__(self, in_places, hidden_places, out_places = 1,  dropout = 0.1, last_dropout = 0.5):
        super(Bottleneck_judge_II,self).__init__()



        self.bottleneck = nn.Sequential(
            nn.Linear(in_places, hidden_places),
            nn.GELU(),
            nn.LayerNorm(hidden_places),
            nn.Linear(hidden_places, hidden_places),
            nn.GELU(),
            nn.LayerNorm(hidden_places),
            nn.Linear(hidden_places, out_places)
        )

        if in_places != out_places:
            self.downsample = nn.Sequential(
                nn.Linear(in_places, out_places)
            )
            
        else:
            self.downsample = nn.Identity()
            

    def forward(self, x):

        out = self.bottleneck(x)
        
        residual = self.downsample(x)
        
        out += residual
        
        return out

class VectorQuantizerEMA(nn.Module):
    def __init__(self, n_embed, dim, beta = None, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        b, s, d = input.shape
        flatten = input.reshape(-1, self.dim)

        with torch.no_grad():
            dist = (
                flatten.pow(2).sum(1, keepdim=True)
                - 2 * flatten @ self.embed
                + self.embed.pow(2).sum(0, keepdim=True)
            )

        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        # print(embed_ind.shape)
        quantize = self.embed_code(embed_ind)
        # print(quantize.shape)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            distri.all_reduce(embed_onehot_sum)
            distri.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)
        """
        diff = (quantize - input.detach()).pow(2).mean() + 0.25*(quantize.detach() - input).pow(2).mean()
        quantize = input + quantize - input.detach()
        """

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize.reshape(b, s, d), diff, embed_ind.unsqueeze(-1)
    
    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class VectorQuantizerEMA_multi_head(nn.Module):
    def __init__(self, n_embed, dim, num_head=4, beta=None, decay=0.99, eps=1e-5, vq_loss_type = 'mae'):
        super().__init__()
        self.dim = dim
        self.n_embed = n_embed
        self.num_head = num_head
        self.decay = decay
        self.eps = eps
        assert dim % num_head == 0, "dim must be divisible by num_head"
        self.head_dim = dim // num_head
 
        # 初始化多头嵌入矩阵 [num_head, head_dim, n_embed]
        embed = torch.randn(num_head, self.head_dim, n_embed, requires_grad = False)*10
        self.register_buffer("embed", embed)
        # 为每个头单独维护cluster_size和embed_avg
        self.register_buffer("cluster_size", torch.zeros(num_head, n_embed))
        self.register_buffer("embed_avg", embed.clone())
        #self.vq_loss_f = F.huber_loss if vq_loss_type == 'huber' else F.mse_loss
        self.vq_loss_f = vq_loss_type
 
    def forward(self, input):
        b, s, d = input.shape
        assert d == self.num_head * self.head_dim, "Input dim mismatch"
        
        # self.embed = self.embed.view(self.num_head,  self.head_dim, self.n_embed)
        
        # 将输入分割到多个头
        input_heads = input.view(b, s, self.num_head, self.head_dim)
        quantize_heads = []
        diff_heads = []
        embed_ind_heads = []
 
        for h in range(self.num_head):
            head_input = input_heads[:, :, h, :].contiguous().view(-1, self.head_dim)
            
            # 计算当前头的距离矩阵
            dist = (
                head_input.pow(2).sum(1, keepdim=True)
                - 2 * head_input @ self.embed[h]
                + self.embed[h].pow(2).sum(0, keepdim=True)
            )
            
            # 量化操作
            _, embed_ind = (-dist).max(1)
            embed_onehot = F.one_hot(embed_ind, self.n_embed).type(head_input.dtype)
            quantize_h = self.embed_code(embed_ind, h).view(b, s, 1, self.head_dim)
            
            # 计算当前头的diff
            # diff_h = (quantize_h.detach() - head_input).pow(2).mean()
            quantize_heads.append(quantize_h)
            # diff_heads.append(diff_h)
            embed_ind_heads.append(embed_ind.view(b, s, 1))
 
        # 合并所有头的结果
        quantize = torch.cat(quantize_heads, dim=2).view(b, s, d)
        # diff = sum(diff_heads) / self.num_head
        embed_ind = torch.cat(embed_ind_heads, dim=2)
 
        # 训练时的EMA更新（按头独立更新）
        if self.training:
            for h in range(self.num_head):
                head_input = input_heads[:, :, h, :].contiguous().view(-1, self.head_dim)
                dist = (
                    head_input.pow(2).sum(1, keepdim=True)
                    - 2 * head_input @ self.embed[h]
                    + self.embed[h].pow(2).sum(0, keepdim=True)
                )
                _, embed_ind = (-dist).max(1)
                embed_onehot = F.one_hot(embed_ind, self.n_embed).type(head_input.dtype)
                
                # 分布式同步统计量
                embed_onehot_sum = embed_onehot.sum(0)
                embed_sum = head_input.t() @ embed_onehot
                distri.all_reduce(embed_onehot_sum)
                distri.all_reduce(embed_sum)
                
                # EMA更新
                self.cluster_size[h].data.mul_(self.decay).add_(
                    embed_onehot_sum, alpha=1 - self.decay
                )
                self.embed_avg[h].data.mul_(self.decay).add_(
                    embed_sum, alpha=1 - self.decay
                )
                
                # 归一化更新
                n = self.cluster_size[h].sum()
                cluster_size = (self.cluster_size[h] + self.eps) / (n + self.n_embed * self.eps) * n
                embed_normalized = self.embed_avg[h] / cluster_size.unsqueeze(0)
                self.embed[h].data.copy_(embed_normalized)
 
        # 直通估计器
        #diff = (quantize.detach() - input).pow(2).mean()
        if self.vq_loss_f == 'huber' or self.vq_loss_f == 'Huber':
            diff = F.huber_loss(input, quantize.detach(), reduction='mean', delta=0.1)
        elif self.vq_loss_f == 'mae' or self.vq_loss_f == 'MAE':
            diff = F.l1_loss(input, quantize.detach())
        else:
            diff = F.mse_loss(input, quantize.detach())
            
        quantize = input + (quantize - input).detach()
        return quantize, diff, embed_ind
    
    def embed_code(self, embed_id, h):
        # 为指定头执行嵌入查找
        return F.embedding(embed_id, self.embed[h].t())

        
class VectorQuantizerEMA_multi_head_ex(nn.Module):
    def __init__(self, n_embed, dim, num_head=4, beta=None, decay=0.99, eps=1e-5, vq_loss_type = 'mae'):
        super().__init__()
        self.dim = dim
        self.n_embed = n_embed
        self.num_head = num_head
        self.decay = decay
        self.eps = eps
        assert dim % num_head == 0, "dim must be divisible by num_head"
        self.head_dim = dim // num_head
 
        # 初始化多头嵌入矩阵 [num_head, head_dim, n_embed]
        embed = torch.randn(num_head, self.head_dim, n_embed, requires_grad = False)*10
        self.register_buffer("embed", embed)
        # 为每个头单独维护cluster_size和embed_avg
        self.register_buffer("cluster_size", torch.zeros(num_head, n_embed))
        self.register_buffer("embed_avg", embed.clone())
        #self.vq_loss_f = F.huber_loss if vq_loss_type == 'huber' else F.mse_loss
        self.vq_loss_f = vq_loss_type
 
    def forward(self, input):
        b, s, d = input.shape
        assert d == self.num_head * self.head_dim, "Input dim mismatch"
        
        # self.embed = self.embed.view(self.num_head,  self.head_dim, self.n_embed)
        
        # 将输入分割到多个头
        input_heads = input.view(b, s, self.num_head, self.head_dim)
        quantize_heads = []
        diff_heads = []
        embed_ind_heads = []
 
        for h in range(self.num_head):
            head_input = input_heads[:, :, h, :].contiguous().view(-1, self.head_dim)
            
            # 计算当前头的距离矩阵
            dist = (
                head_input.pow(2).sum(1, keepdim=True)
                - 2 * head_input @ self.embed[h]
                + self.embed[h].pow(2).sum(0, keepdim=True)
            )
            
            # 量化操作
            _, embed_ind = (-dist).max(1)
            embed_onehot = F.one_hot(embed_ind, self.n_embed).type(head_input.dtype)
            quantize_h = self.embed_code(embed_ind, h).view(b, s, 1, self.head_dim)
            
            # 计算当前头的diff
            # diff_h = (quantize_h.detach() - head_input).pow(2).mean()
            quantize_heads.append(quantize_h)
            # diff_heads.append(diff_h)
            embed_ind_heads.append(embed_ind.view(b, s, 1))
 
        # 合并所有头的结果
        quantize = torch.cat(quantize_heads, dim=2).view(b, s, d)
        # diff = sum(diff_heads) / self.num_head
        embed_ind = torch.cat(embed_ind_heads, dim=2)
 
        # 训练时的EMA更新（按头独立更新）
        if self.training and self.decay < 1:
            for h in range(self.num_head):
                head_input = input_heads[:, :, h, :].contiguous().view(-1, self.head_dim)
                dist = (
                    head_input.pow(2).sum(1, keepdim=True)
                    - 2 * head_input @ self.embed[h]
                    + self.embed[h].pow(2).sum(0, keepdim=True)
                )
                dist = F.gumbel_softmax(dist, tau = 1.6, dim = -1) 
                _, embed_ind = (-dist).max(1)
                embed_onehot = F.one_hot(embed_ind, self.n_embed).type(head_input.dtype)
                
                # 分布式同步统计量
                embed_onehot_sum = embed_onehot.sum(0)
                embed_sum = head_input.t() @ embed_onehot
                distri.all_reduce(embed_onehot_sum)
                distri.all_reduce(embed_sum)
                
                # EMA更新
                self.cluster_size[h].data.mul_(self.decay).add_(
                    embed_onehot_sum, alpha=1 - self.decay
                )
                self.embed_avg[h].data.mul_(self.decay).add_(
                    embed_sum, alpha=1 - self.decay
                )
                
                # 归一化更新
                n = self.cluster_size[h].sum()
                cluster_size = (self.cluster_size[h] + self.eps) / (n + self.n_embed * self.eps) * n
                embed_normalized = self.embed_avg[h] / cluster_size.unsqueeze(0)
                self.embed[h].data.copy_(embed_normalized)
 
        # 直通估计器
        #diff = (quantize.detach() - input).pow(2).mean()
        if self.vq_loss_f == 'huber' or self.vq_loss_f == 'Huber':
            diff = F.huber_loss(input, quantize.detach(), reduction='mean', delta=0.1)
        elif self.vq_loss_f == 'mae' or self.vq_loss_f == 'MAE':
            diff = F.l1_loss(input, quantize.detach())
        else:
            diff = F.mse_loss(input, quantize.detach())
            
        quantize = input + (quantize - input).detach()
        return quantize, diff, embed_ind
    
    def embed_code(self, embed_id, h):
        # 为指定头执行嵌入查找
        return F.embedding(embed_id, self.embed[h].t())

class VectorQuantizerEMA_multi_head_revival(nn.Module):
    def __init__(self, n_embed, dim, num_head=4, beta=None, decay=0.99, eps=1e-5, vq_loss_type='mae'):
        super().__init__()
        self.dim = dim
        self.n_embed = n_embed
        self.num_head = num_head
        self.decay = decay
        self.eps = eps
        assert dim % num_head == 0, "dim must be divisible by num_head"
        self.head_dim = dim // num_head

        embed = F.normalize(torch.randn(num_head, self.head_dim, n_embed, requires_grad = False), p=2, dim=1)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(num_head, n_embed))
        self.register_buffer("embed_avg", embed.clone())
        self.vq_loss_f = vq_loss_type

        # ✅ 新增：复活超参
        self.revival_interval = 1
        self.dead_threshold = 1.0
        self.register_buffer("step_count", torch.tensor(0, dtype=torch.long))

    # ✅ 新增：复活函数
    @torch.no_grad()
    def _revive_dead_codes(self, input_heads):
        B, S, _, _ = input_heads.shape
        for h in range(self.num_head):
            dead_ids = (self.cluster_size[h] < self.dead_threshold).nonzero(as_tuple=True)[0]
            n_dead = dead_ids.numel()
            if n_dead == 0:
                continue
    
            head_input = input_heads[:, :, h, :].reshape(-1, self.head_dim)   # [N, head_dim]
            n_avail = head_input.size(0)
    
            # 不重复采样：最多复活 n_avail 个
            n_resample = min(n_dead, n_avail)
            rand_idx = torch.randperm(n_avail, device=head_input.device)[:n_resample]
            revive_vec = head_input[rand_idx]                # [n_resample, head_dim]
    
            # 只替换前 n_resample 个死亡码
            self.embed[h][:, dead_ids[:n_resample]] = revive_vec.t()
            self.cluster_size[h][dead_ids[:n_resample]] = self.dead_threshold
            self.embed_avg[h][:, dead_ids[:n_resample]] = revive_vec.t() * self.dead_threshold

    # ------------ 你原来的 forward 一字未动 ------------
    def forward(self, input):
        b, s, d = input.shape
        assert d == self.num_head * self.head_dim, "Input dim mismatch"
        input_heads = input.view(b, s, self.num_head, self.head_dim)
        quantize_heads, embed_ind_heads = [], []

        for h in range(self.num_head):
            head_input = input_heads[:, :, h, :].contiguous().view(-1, self.head_dim)
            dist = (
                head_input.pow(2).sum(1, keepdim=True)
                - 2 * head_input @ self.embed[h]
                + self.embed[h].pow(2).sum(0, keepdim=True)
            )
            _, embed_ind = (-dist).max(1)
            embed_onehot = F.one_hot(embed_ind, self.n_embed).type(head_input.dtype)
            quantize_h = F.embedding(embed_ind, self.embed[h].t()).view(b, s, 1, self.head_dim)
            quantize_heads.append(quantize_h)
            embed_ind_heads.append(embed_ind.view(b, s, 1))

        quantize = torch.cat(quantize_heads, dim=2).view(b, s, d)
        embed_ind = torch.cat(embed_ind_heads, dim=2)

        if self.training and self.decay < 1:
            # ✅ 新增：复活检查
            self.step_count += 1
            if self.training and (self.step_count % self.revival_interval) == 0:
                self._revive_dead_codes(input_heads)
                
            for h in range(self.num_head):
                head_input = input_heads[:, :, h, :].contiguous().view(-1, self.head_dim)
                dist = (
                    head_input.pow(2).sum(1, keepdim=True)
                    - 2 * head_input @ self.embed[h]
                    + self.embed[h].pow(2).sum(0, keepdim=True)
                )
                dist = F.gumbel_softmax(dist, tau=1.6, dim=-1)
                _, embed_ind = (-dist).max(1)
                embed_onehot = F.one_hot(embed_ind, self.n_embed).type(head_input.dtype)
                embed_onehot_sum = embed_onehot.sum(0)
                embed_sum = head_input.t() @ embed_onehot
                distri.all_reduce(embed_onehot_sum)
                distri.all_reduce(embed_sum)
                self.cluster_size[h].data.mul_(self.decay).add_(embed_onehot_sum, alpha=1 - self.decay)
                self.embed_avg[h].data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
                n = self.cluster_size[h].sum()
                cluster_size = (self.cluster_size[h] + self.eps) / (n + self.n_embed * self.eps) * n
                embed_normalized = self.embed_avg[h] / cluster_size.unsqueeze(0)
                self.embed[h].data.copy_(embed_normalized)
        if self.vq_loss_f == 'huber' or self.vq_loss_f == 'Huber':
            diff = F.huber_loss(input, quantize.detach(), reduction='mean', delta=0.1)
        elif self.vq_loss_f == 'mae' or self.vq_loss_f == 'MAE':
            diff = F.l1_loss(input, quantize.detach())
        else:
            diff = F.mse_loss(input, quantize.detach())
        quantize = input + (quantize - input).detach()
        return quantize, diff, embed_ind

    def embed_code(self, embed_id, h):
        return F.embedding(embed_id, self.embed[h].t())

class VectorQuantizer(nn.Module):

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embed = nn.Embedding(self.K, self.D)
        # self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents):
        # latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BHW x D]

        # Compute L2 distance between latents and embedding weights
        
        with torch.no_grad():
            dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
                   torch.sum(self.embed.weight ** 2, dim=1) - \
                   2 * torch.matmul(flat_latents, self.embed.weight.t())  # [BHW x K]
                   
        # if self.training:
        #     dist = F.gumbel_softmax(dist, dim = 1)

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embed.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        # return quantized_latents.permute(0, 3, 1, 2).contiguous(), vq_loss  # [B x D x H x W]
        return quantized_latents, vq_loss, encoding_inds.view(latents_shape[:-1], 1)
    
    def regression_to_latents(self, regression):
        
    
        b, s, d = regression.shape
        
        regression_index = F.one_hot(regression.argmax(dim = -1), self.K)
        
        # print(regression_index.shape)
        
        quantized_latents = torch.matmul(regression_index.float(), self.embedding.weight)
        
        # print(quantized_latents.shape, self.embedding.weight.shape)
        
        return quantized_latents#.reshape(b, s, self.D)


def print_frozen_names_only(model):
    """极简版：只打印冻结的模块名"""
    base_model = model.module if hasattr(model, 'module') else model
    
    frozen_names = []
    for name, module in base_model.named_modules():
        params = list(module.parameters(recurse=False))
        if not params:
            continue
        if all(not p.requires_grad for p in params):
            frozen_names.append(name)
    
    if frozen_names:
        print(f"冻结模块 ({len(frozen_names)} 个):")
        for name in frozen_names:
            print(f"  - {name}")
    else:
        print("无冻结模块")
    
    return frozen_names



class GracefulSaver:
    def __init__(self, model, optimizer, prefix="ckpt", save_dir="./"):
        self.model = model
        self.optimizer = optimizer
        self.prefix = prefix
        self.save_dir = save_dir
        self.interrupted = False
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存原始处理程序以便恢复
        self.original_handler = signal.signal(signal.SIGINT, self._handler)
    
    def _handler(self, signum, frame):
        self.interrupted = True
        print(f"\nCtrl+C pressed. Saving checkpoint...")
        self.save()  # 去掉时间戳，直接覆盖保存
        print("Checkpoint saved. Exiting...")
        exit(0)
    
    def save(self):
        """保存检查点，自动处理 DDP"""
        # 解包 DDP（如果是的话）
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        
        model_path = os.path.join(self.save_dir, f"{self.prefix}_model.pt")
        opt_path = os.path.join(self.save_dir, f"{self.prefix}_optimizer.pt")
        
        torch.save(model_to_save.state_dict(), model_path)
        torch.save(self.optimizer.state_dict(), opt_path)
        print(f"Saved: {model_path} & {opt_path}")
    
    def restore_handler(self):
        """恢复原始信号处理（如需要）"""
        signal.signal(signal.SIGINT, self.original_handler)


        
def print_lr_dict(optimizer, model=None):
    """
    打印优化器学习率
    可选传入 model 用于显示模块名（会自动解包 DDP）
    """
    # 解包 DDP
    if model is not None:
        model = model.module if hasattr(model, 'module') else model
    
    print("-" * 60)
    for i, g in enumerate(optimizer.param_groups):
        lr = g['lr']
        wd = g.get('weight_decay', 0)
        n_params = len(g['params'])
        n_elements = sum(p.numel() for p in g['params'])
        
        # 尝试获取组名（如果传入了 model）
        group_name = f"group_{i}"
        if model is not None and g['params']:
            # 查找参数属于哪个模块
            first_param = g['params'][0]
            for name, module in model.named_modules():
                if any(p is first_param for p in module.parameters(recurse=False)):
                    group_name = name
                    break
        
        print(f"{group_name:30s} | lr={lr:.2e} | wd={wd:.2e} | params={n_elements:,}")
    print("-" * 60)
    
    
