#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 01:54:58 2024

@author: yuanbeiming
"""
import torch
from torch import einsum
import torch.nn as nn
from einops import rearrange, repeat
import math
import numpy as np
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from Blocks_clip import *

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x,**kwargs):
        return self.fn(self.norm(x), **kwargs)
    
    
    
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
    
    
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 32, dropout = 0.):
        #dim_head: dim of signle head
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim = -1)
        self.to_out = nn.Identity()
 
 
    def forward(self, q, k, v, **kwargs):
        h = self.heads
        
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        # out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
    

    
class Linear_Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 32, dropout = 0.):
        #dim_head: dim of signle head
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.to_out = nn.Identity()
        self.beta = nn.Parameter(torch.ones(1)*4.6)
        
        
    def sigma(self, x):
        
        return F.elu(x) + 1.
 
    def forward(self, q, k, v, M, Z, **kwargs):
        """
        d_key = d
        
        d_value = v
        
        d_model_k = n
        
        d_model_q = m
        
        k: b h n d
        
        q: b h m d
        
        v: b h n v
        
        M: b h d v
        
        Z: b h d 1
        
        """
        h = self.heads
        sigma_k = self.sigma(k)
        sigma_q = self.sigma(q)
        # print(sigma_q.shape, M.shape)
        A_mem = einsum('b h m d, b h d v-> b h m v', sigma_q, M)/einsum('b h m d, b h d i -> b h m i', sigma_q, Z)#b h m v
        # A_mem = torch.matmul(sigma_q, M)/torch.matmul(sigma_q, Z)#b h m v
        moment = einsum('b h n d, b h d v -> b h n v', sigma_k, M)/einsum('b h n d, b h d i -> b h n i', sigma_k, Z)#b h n v
        # moment = torch.matmul(sigma_k, M)/torch.matmul(sigma_k, Z)
        # print(k.shape)
        # print(Z.shape)
        if self.training:
            beta = F.sigmoid(self.beta).clamp(min = 0.9, max = 0.999)
            Z = beta*Z + sigma_k.sum(dim = -2).unsqueeze(-1)
            M = beta*M + einsum('b h n d, b h n v -> b h d v', sigma_k, (v - moment))
        """
        Z = Z + sigma_k.sum(dim = -2).unsqueeze(-1)
        M = M + einsum('b h n d, b h n v -> b h d v', sigma_k, (v - moment))
        """
        return self.to_out(A_mem), M, Z
    
    
class Infinity_Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 32, dropout = 0.):
        #dim_head: dim of signle head
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
 
 
        self.heads = heads
        self.scale = dim_head ** -0.5
 
 
        # self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        
        
        self.Linear_Attention = Linear_Attention(dim, heads, dim_head, dropout)
        
        self.dot_Attention = Attention(dim, heads, dim_head, dropout)
        
        self.beta = nn.Parameter(torch.randn(1, self.heads, 1, 1))
 
 
        self.to_out = nn.Identity()
        
        
    def sigma(self, x):
        
        return F.elu(x) + 1.
 
    def forward(self, x, M, Z, **kwargs):
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v= map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))
        # print(q.shape, k.shape, v.shape)
        # q, k, v, M, Z = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v, M, Z))
        A_mem, M, Z = self.Linear_Attention(q, k, v, M, Z)
        A_dot = self.dot_Attention(q, k, v)
        beta = F.sigmoid(self.beta)
        A = beta*A_mem + (1-beta)*A_dot
        A = rearrange(A, 'b h n d -> b n (h d)', h = self.heads)
        return self.to_out(A), M, Z
    
    

    
class Infinity_transformer_layers(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.depth = depth
        for _ in range(depth):#L
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                Infinity_Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, M, Z):
        
        # M_next = []
        # Z_next = []
        for index, (norm, attn, ff) in enumerate(self.layers):
            x = norm(x)
            x_, M[index], Z[index] = attn(x, M[index], Z[index], name = 'xyk', name_didi = 'vql')
            x = x_ + x
            # print(x.shape)
            x = ff(x) + x
            
            x = x + torch.randn_like(x)

        return x, M, Z
    
    
class Infinity_Transformer(nn.Module):
    def __init__(self, words, dim, depth, heads, dim_head, mlp_dim, num_cls = 0, dropout = 0., PositionalEncoding = True):
        super().__init__()
        
        
        self.PositionalEncoding = PositionalEncoding
        if  self.PositionalEncoding:
            self.pos_embedding = nn.Parameter(torch.randn(1, words + num_cls, dim))
            
        
        self.num_cls = num_cls
        if num_cls != 0:
            self.cls_token = nn.Parameter(torch.randn(1, num_cls, dim))
        self.transformer = Infinity_transformer_layers(dim, depth, heads, dim_head, mlp_dim, dropout = 0.)
        self.dropout = nn.Dropout(dropout)
        
        self.depth = depth
        
    def forward(self, x, M, Z):
        b,n,d = x.shape
        
        
        
        if self.num_cls != 0:
            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
            x = torch.cat((cls_tokens, x), dim=1)
            
        if  self.PositionalEncoding:
            x += .5*self.pos_embedding[:, :] + .5*self.pos_embedding[:, [0,3,6,1,4,7,2,5,8] + [i for i in range(9, n)]]
        # dropout
        x = self.dropout(x)

        x, M, Z = self.transformer(x, M, Z)

        return x, M, Z
    
    
if __name__ == '__main__':
    
    x = torch.randn(10, 5, 8)
    
    
    depth = 6
    
    M = torch.randn(depth, 2, 4, 4)
    
    Z = torch.randn(depth, 2, 4, 1)
    
    M = [M[[i]].expand(10, -1,-1,-1) for i in range(depth)]
    Z = [Z[[i]].expand(10, -1,-1,-1) for i in range(depth)]
    
    # AA = Infinity_Attention(dim = 8, heads = 2, dim_head = 4, dropout = 0.)
    
    
    # out = AA(x, M, Z)
    
    AT = Infinity_Transformer(words = 5, dim = 8, depth = depth, heads = 2, dim_head = 4,mlp_dim = 8,  dropout = 0.)
    
    
    out_ = AT(x, M, Z)