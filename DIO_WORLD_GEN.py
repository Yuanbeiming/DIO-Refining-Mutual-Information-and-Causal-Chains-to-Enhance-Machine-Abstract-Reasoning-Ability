#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 11:26:31 2021

@author: yuanbeiming
"""
import torch
import torch.nn as nn
import numpy as np


import torch.nn.functional as F

from Blocks_clip import *

from torch.nn.utils import spectral_norm


from einops.layers.torch import Rearrange
#from SinkhornDistance import SinkhornDistance
from torchvision import transforms

t = transforms.Resize((80,80))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
big = False
dropout = False
temperature = 1e-6




class Sigmoid_up(nn.Module):
    def __init__(self, alpha = 10, step_size = 0.01):
        super().__init__()

        
    def forward(self, x):
        
        
        return (torch.sigmoid(x) +1) /2
    

class Sigmoid_down(nn.Module):
    def __init__(self, alpha = 10, step_size = 0.01):
        super().__init__()

        
    def forward(self, x):
        
        
        return torch.sigmoid(x) /2


class Recat(nn.Module):
    def __init__(self, num_aux_candidates=8):
        super(Recat, self).__init__()
        self.num_candidates = num_aux_candidates + 8
        
    def forward(self, x):
        b, n, s, d = x.shape
      
        
        indices = []
        # 1. 基础 3x3 矩阵 (0-8)
        indices += [0,1,2,3,4,5,6,7,8]
        

        base_idx = 9  # 候选起始索引
        for i in range(self.num_candidates-1):
            indices += [6, 7, base_idx + i]
            
        # 3. 列方向基础: [0,3,6], [1,4,7], [2,5,8]
        indices += [0,3,6,1,4,7,2,5,8]
        

        for i in range(self.num_candidates-1):
            indices += [2, 5, base_idx + i]
        #print('indices1:', indices)
        

        
        return x[:, indices].reshape(b, -1, 3, s, d)
        
    
class Recombine(nn.Module):
    def __init__(self, num_aux_candidates=10):
        super(Recombine, self).__init__()
        
        self.num_candidates = num_aux_candidates + 8
        
        # self.m = (2+num_aux_candidates)*2

    def forward(self, x):
        b ,s ,m ,d = x.shape
        
        indices = []
        
        
        cr = int(m/2)
        
        assert self.num_candidates == cr-2

        for i in range(2, cr):
            indices += [0,1,i, cr, cr + 1, cr + i]
        #print('indices2:', indices)
       
        
        return x[:,:, indices].reshape(b, s, cr-2, 6, d)

    
    
    
    
class print_layer(nn.Module):
    def __init__(self):
        super(print_layer, self).__init__()
       

    def forward(self, x):
        
        print(x.shape)

        
        return x


class shuffle_sample(nn.Module):
    def __init__(self):
        super(shuffle_sample, self).__init__()
        
    def forward(self, x):
        #"""
        if self.training:
        
            s = x.shape[-2]
            
            assert s == 4
            
            index = torch.randperm(s)
            
            
            return x[:,:,:,:,index]
        
        else:
        #"""
            return x
    
    
class shuffle_sample_(nn.Module):
    def __init__(self):
        super(shuffle_sample_, self).__init__()
        
    def forward(self, x):
        #"""
        if self.training:
        
            s = x.shape[-2]
            
            # assert s == 8
            
            index = torch.randperm(s)
            
            
            return x[:,index]
        
        else:
        #"""
            return x


class get_choise(nn.Module):
    def __init__(self):
        super(get_choise, self).__init__()
       

    def forward(self, x):
        
        b, s, n, m, d = x.shape
        
        assert m == 6

        
        return torch.stack([
                x[:,:,:,[0,1,3,4]],
                x[:,:,:,[0,1,3,5]],
                x[:,:,:,[0,1,4,5]],
                x[:,:,:,[0,2,3,4]],
                x[:,:,:,[0,2,3,5]],
                x[:,:,:,[0,2,4,5]],
                x[:,:,:,[1,2,3,4]],
                x[:,:,:,[1,2,3,5]],
                x[:,:,:,[1,2,4,5]],
                ], dim = 3)

class Cross_Transformer_layer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):#L
            self.layers.append(nn.ModuleList([
                
                Cross_Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)),
                
                # Cross_Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                # PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)),
                
                
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, q_1, kv):
        
        # print(q.shape)
        for c_attn, ff, attn, ff_1 in self.layers:

            kv = attn(kv, name = 'yuanbeiming', name_didi = 'chendiancheng') + kv
            kv = ff_1(kv) + kv
            
            q_1 = c_attn(q_1, kv) + q_1
            q_1 = ff(q_1) + q_1

        return q_1
    


class Cross_Transformer(nn.Module):
    def __init__(self, words, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super(Cross_Transformer,self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, words, dim))
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Cross_Transformer_layer(dim, depth, heads, dim_head, mlp_dim, dropout = 0.)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q_1, kv):
        b,n,d = kv.shape
        kv = kv + self.pos_embedding[:, :]
        # dropout
        kv = self.dropout(kv)

        x = self.transformer(q_1, kv)

        return x

        


class To_image(nn.Module):
    def __init__(self, latent_dim = 1280):
        super().__init__()
        
        in_feature = 6400
        
        self.lin = nn.Sequential(Rearrange('b c h w -> b (c h w)'),
                                 nn.Linear(in_feature, latent_dim),
                                 nn.BatchNorm1d(latent_dim),
                                 nn.GELU(),
                                 nn.Linear(latent_dim, in_feature),
                                 Rearrange('b (c h w) -> b c h w', c = 1, h = 80, w = 80)
            
            )


        
    def forward(self, x):
        out = self.lin(x)
        
        return out + x
    
""" 
class To_image_cnn(nn.Module):
    def __init__(self, in_ch=1, hidden_ch=4, gn_ch = 4):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_ch, hidden_ch, 3, padding=1)
        self.gn = nn.GroupNorm(gn_ch, hidden_ch)  # 16%8=0
        self.conv2 = nn.Conv2d(hidden_ch + in_ch, in_ch, 3, padding=1)
        
        # 全零初始化
        nn.init.zeros_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.gn(out)
        out = F.gelu(out)
        out = torch.cat([x, out], dim=1)
        out = self.conv2(out)
        
        return out + x
        
"""    
class To_image_cnn(nn.Module):
    def __init__(self, in_ch=1, hidden_ch=4, gn_ch = 4):
        super().__init__()
        
        # 下采样: 80x80 -> 40x40
        self.conv1 = nn.Conv2d(in_ch, hidden_ch, 4, stride=2, padding=1)
        self.gn1 = nn.GroupNorm(gn_ch, hidden_ch)
        
        # 40x40 空间处理
        self.conv2 = nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1)
        self.gn2 = nn.GroupNorm(gn_ch, hidden_ch)
        
        # 上采样: 40x40 -> 80x80
        self.conv3 = nn.ConvTranspose2d(hidden_ch, hidden_ch, 4, stride=2, padding=1)
        self.gn3 = nn.GroupNorm(gn_ch, hidden_ch)
        
        # 输出层，融合原始输入
        self.conv4 = nn.Conv2d(hidden_ch + in_ch, in_ch, 3, padding=1)
        
        # 输出层初始化为零（近似恒等）
        nn.init.zeros_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)
        nn.init.zeros_(self.conv4.weight)
        nn.init.zeros_(self.conv4.bias)
        
    def forward(self, x):
        identity = x
        
        # 编码路径
        h = self.conv1(x)
        h = self.gn1(h)
        h = F.gelu(h)
        
        h = self.conv2(h)
        h = self.gn2(h)
        h = F.gelu(h)
        
        # 解码上采样
        h = self.conv3(h)
        h = self.gn3(h)
        h = F.gelu(h)
        
        # 融合原始输入，残差输出
        h = torch.cat([h, identity], dim=1)
        out = self.conv4(h)
        
        return out + identity
   

class raven_clip(nn.Module):
    def __init__(self, *args, num_aux_candidates = 8):
        super(raven_clip,self).__init__()

        

        

        size = 80
        patch = 20
        
        if big:
            num_head = 8
            num_depth = 6
            self.low_dim = 256
        else:
            num_head = 4
            num_depth = 3
            self.low_dim = 128

        
    
            
        if dropout:
            _dropout = 0.1
        else:
            _dropout = 0

        self.beta = 0.25
        txt_data = []
        for c in range(8,14):#color
            for n in range(8,14):#number
                for p in range(8,14):#position
                    for s in range(8,14):#size
                        for t in range(8,14):#type
                            txt_data.append(np.array([ 3,  c,  4, n,  5, p,  6, s, 7, t]))
                        
        



        
        txt_data = np.array(txt_data)[:-1]
        
        assert txt_data.shape[0] == 7775
        
        txt_size = 7775
        
        
        self.dict_ = { 'EOF':0,
                      'shape':1,
                      'line':2, 
                      'color':3, 
                      'number':4, 
                      'position':5, 
                      'size':6, 
                      'type':7, 
                      'progression':8, 
                      'XOR':9, 
                      'OR':10, 
                      'AND':11, 
                      'consistent_union':12,
                      ' NA':13}
        
        self.txt_data = torch.from_numpy(txt_data[None,:,:]).long()
        
        
        assert self.txt_data.shape[1] == 7775
        
        self.txt_data.requires_grad = False
    

        self.w = int(size/patch)*int(size/patch)

        #self.num_embeddings = 8192
        self.num_embeddings = (2**14)#*self.w
        
        self.num_aux_candidates = num_aux_candidates #int((16**vql_heads)/self.w)


        
        num_candidates = num_aux_candidates + 8
        
        m = (2+num_candidates)*2


        self.temperature = temperature

        
        self.num_rule = 4

        is_mix = True

        is_cnn = False

        self.is_dou = False 
        self.if_sn = True

        self.vql_heads = vql_heads = 2

        self.name = 'DIO_WORLD_'+ ('sn_' if self.if_sn else '_') + 'toxic_embdv'+str(self.num_embeddings)+ '_heads_' +str(vql_heads) + '_candidates_' +str(num_aux_candidates)

        if self.is_dou: self.name = self.name + '_double_decoder'

        if is_cnn: self.name = self.name + '_add_CNN'

        if is_mix == False: self.name = self.name + '_without_mix'
            
        self.vit = nn.Sequential(ViT(image_size = size, 
                                   patch_size = patch,  
                                   dim = self.low_dim*2,
                                   depth = num_depth, 
                                   heads = num_head, 
                                   mlp_dim = self.low_dim*2,
                                   channels = 1, 
                                   dim_head = int(self.low_dim*2/(num_head)), 
                                   dropout = _dropout, 
                                   emb_dropout = _dropout),
    
                                 )
        
        self.recat = nn.Sequential(Recat(num_aux_candidates = num_aux_candidates),
        Rearrange('b m n s d -> b s m (n d)', s = self.w, n = 3, m = m),
        

        )#b*16, 16, dimS
        
        


        num_decoder_depth = num_depth*2+1
          
        if self.is_dou:
            self.decoder_up = nn.Sequential(Rearrange('b n s d -> (b n) s d',  s = self.w),
                                              ViT_reverse(#words = self.w, 
                                                                    image_size = 80,  
                                                                   
                                                                    patch_size = 20,
                                                                   
                                                                    channels = 1,
                                                                   
                                                                    dim = self.low_dim, 
                                                                   
                                                                    depth = num_decoder_depth,
                                                                    
                                                                    heads = num_head,
                                                                    
                                                                    mlp_dim = self.low_dim,
                                                                    
                                                                    dim_head = int(self.low_dim/num_head),
                                                                    
                                              ), 
                                              To_image(640) if is_mix else nn.Identity(),
                                              #To_image(640) if is_mix else nn.Identity(),
                                              To_image_cnn() if is_cnn else nn.Identity(),
                                              Sigmoid_up()
                                              )
        
        self.decoder_down = nn.Sequential(Rearrange('b n s d -> (b n) s d',  s = self.w),
                                              ViT_reverse(#words = self.w, 
                                                                    image_size = 80,  
                                                                   
                                                                    patch_size = 20,
                                                                   
                                                                    channels = 1,
                                                                   
                                                                    dim = self.low_dim, 
                                                                   
                                                                    depth = num_decoder_depth,
                                                                    
                                                                    heads = num_head,
                                                                    
                                                                    mlp_dim = self.low_dim,
                                                                    
                                                                    dim_head = int(self.low_dim/num_head),
                                                                
                                                ),
                                                To_image(1280) if is_mix else nn.Identity(),
                                                #To_image(1280) if is_mix else nn.Identity(),
                                                To_image_cnn(hidden_ch = 4, gn_ch = 4) if is_cnn else nn.Identity(),
                                                Sigmoid_down() if self.is_dou else nn.Sigmoid()
                                                
                                                )
        
        
        
        
        self.g_function = nn.Sequential(
            Rearrange('b s m d -> (b s m) d'),

            Bottleneck_judge(3*self.low_dim, int(self.low_dim*0.75), 3*self.low_dim),#10,10
            
            Bottleneck_judge(3*self.low_dim, int(self.low_dim*0.75), 3*self.low_dim),#10,10
            
            Bottleneck_judge(3*self.low_dim, int(self.low_dim*0.75), self.low_dim),#10,10
            
            Rearrange('(b s m) d -> b s m d', s = self.w, m = m),

            

            
        )
        


        self.graph_clip = nn.Sequential( 
                                        nn.Sequential(get_choise(), shuffle_sample(),),
                                        Rearrange('b s n c m d -> (b s n c) m d', m = 4, n = num_candidates, c = 9),
                                        graph_mask_transformer(words = 4, dim = self.low_dim, depth = num_depth, heads = num_head, dim_head = int(self.low_dim/num_head), mlp_dim = self.low_dim, num_cls = self.num_rule, dropout = 0.1),
                                        take_cls(self.num_rule, keepdim = True),
                                        Rearrange('(b s n c) m d -> b s n c m d', s = self.w, d = self.low_dim, m = self.num_rule, n = num_candidates, c = 9),
                                        )

        
        self.recombine = Recombine(num_aux_candidates = num_aux_candidates)
        self.shuffle = shuffle_sample_()
        self.rearrange = Rearrange('b s n c d -> (b s n) c d', n = num_candidates, s = self.w)
                                      
        
        self.cross_graph_clip = Cross_Transformer(words = 8, dim = self.low_dim, 
                                                  depth = num_depth, heads = num_head, 
                                                  dim_head = int(self.low_dim/num_head), 
                                                  mlp_dim = self.low_dim, dropout = 0.1)
        
        self.to_out = nn.Sequential( 
                              
                                        Rearrange('(b s n) c d -> b s n c d', s = self.w, n = num_candidates, c = 1),
                                        
                                        Rearrange('b s n c d -> b n s c d', s = self.w, n = num_candidates, c = 1),

                                        )


        
        self.tajador = nn.Sequential(Rearrange('b m s n d -> (b m s n) d'),
                            Bottleneck_judge(self.low_dim, self.low_dim),
                            )
        

        self.is_gumbel_vql = False
        gumbel_tau = 5.0
        init_length = 12
        self.vql = VectorQuantizerEMA_multi_head_revival_with_recored(n_embed = self.num_embeddings,
                                              dim = self.low_dim,
                                              num_head = vql_heads,
                                              init_length = init_length,
                                              beta = self.beta,
                                              decay = 1,
                                              #decay = 1 - (1e-8),
                                              #decay = 0.99,
                                              #decay = 0.9,
                                              #decay = 0.99999,
                                              vq_loss_type = 'mse',
                                              num_positions = self.w,
                                              MAX_sample = 1920e4,
                                              is_gumbel = self.is_gumbel_vql,
                                              gumbel_tau = gumbel_tau,
                                              if_revival = True,
                                              normal_init = False,
                                              is_gumbel_update = False
                                              )
        if self.if_sn: self._add_spectral_norm()
        self.replace_x = 0
        self.max_replace = int(self.w/2)
        self.reasoning_noise = 1.0
        self.option_noise = 0.05
        self.noise_x = init_length/(self.low_dim/vql_heads)**0.5 #if is_cnn else 2.5
        self.noise_y = 6.0
        self.s_x = 0 if self.vql.decay == 1 else 0
        self.pattern_extent = 1000000
        
        self.vq_extent = 1.3e6
        self.vq_extent_2 = 1e4 
        self.vq_lower_bound = 1.69
        self.vq_lower_bound_2 = 1.
        
        self.reasoning_extent = 400000  
        self.dio_delta = 1e-7
        
        self.gen_extent = 3e6 
        
        self.norm_extent = 1e7  
        self.bias_extent = 1e2
        self.norm_lower_bound = init_length
        
        self.noise_pattern = self.noise_x != self.reasoning_noise
        self.noise_w = torch.distributions.Uniform(min(self.reasoning_noise, 1), self.noise_x) if self.noise_x != self.reasoning_noise else None
        self.noise_v = torch.distributions.Uniform(min(self.reasoning_noise, 1), self.noise_y) if self.noise_y != self.reasoning_noise else None
        self.toxic = True
        self.j_mean = False
        self.only_option_noise = False
        self.print_norm = False #self.vql.decay == 1
        self.strike_norm = True

        if self.j_mean == False: self.j_position = nn.Parameter(torch.zeros(self.w, self.low_dim))
        
        


        
        

        if self.vql.decay < 1:
            print('val_decay:', self.vql.decay)
        else:
            print('*'*100)
            print('Worning! embed is fixed')
        print('num_aux_candidates:', num_aux_candidates)
        print('=' * 60)
        print(f"reasoning_noise is {self.reasoning_noise}")
        print(f"noise_x is {self.noise_x}")
        print(f"noise_y is {self.noise_y}")
        print(f"s_x is {self.s_x}")
        print(f"vq_extent is {self.vq_extent}")
        print(f"vq_lower_bound is {self.vq_lower_bound}")
        print(f"vq_extent_2 is {self.vq_extent_2}")
        print(f"vq_lower_bound_2 is {self.vq_lower_bound_2}")
        print(f"reasoning_extent is {self.reasoning_extent}")
        print(f"gen_extent is {self.gen_extent}")
        print(f"dio_delta is {self.dio_delta}")
        print(f"norm_extent is {self.norm_extent}")
        print(f"norm_lower_bound is {self.norm_lower_bound}")
        print(f"noise_pattern is {self.noise_pattern}")
        print(f"noise_w is {self.noise_w}")
        print(f"noise_v is {self.noise_v}")
        print(f"toxic is {self.toxic}")
        print(f"j_mean is {self.j_mean}")
        print(f"init_length is {init_length}")
        print(f"only_option_noise is {self.only_option_noise}")
        if self.only_option_noise: print(f"option_noise is {self.option_noise}")
        print(f"strike_norm is {self.strike_norm}")
        print('=' * 60)
        

    def _add_spectral_norm(self):
        
        
        exclude_names = {'vit', 'decoder_up', 'decoder_down', 'txt_clip'}
        
        for name, module in self.named_modules():
            # 跳过模型本身（name为空）和排除模块及其子模块
            if not name:
                continue
            
            # 检查路径中是否包含排除的模块名（如 vit.transformer 会被排除）
            if any(excluded in name.split('.') for excluded in exclude_names):
                continue
            
            # 只为 Linear 和 Conv2d 添加谱归一化
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                spectral_norm(module)
                print(f'add sn to: {name}')
                
    def eval_except_vql(self):
	    """
	    将除 vql 及其子模块外的所有模块置为 eval()，
	    vql 保持 train() 模式（用于 EMA 更新等）。
	    """
	    for name, module in self.named_modules():
	        if name == 'vql' or name.startswith('vql.'):
	            module.train()
	        else:
	            module.eval()
	    return self
	    

       
    
    def forward_cross_attention(self,qkv):
        
       # print(q_1.shape)
       
       q, kv = qkv.split([1,8], dim = 3)
       
       q = self.rearrange(q)
       
       kv = self.rearrange(kv)
       
       kv = self.shuffle(kv)
       
       out = self.cross_graph_clip(q, kv)
       
       
       
       return self.to_out(out)
       
       
       
    
    
    def forward(self, x):
 
        
        b, n, h, w = x.shape
        

        state = x = x.view(b*n, 1, h, w)
        
        x = self.vit(x)
        
        x, bias = x.chunk(2, dim = -1)

        norm = x.reshape(b*n, self.w, self.vql_heads, -1).norm(dim = -1)
        

        if self.strike_norm : x = (x.reshape(b*n, self.w, self.vql_heads, -1)/norm.detach()[:,:,:,None] * self.norm_lower_bound).reshape(b*n, self.w, -1)


        x_recon, vq_loss, _ = self.vql(x)
 
        x_recon = x_recon.view(b, n, self.w, self.low_dim)
        
        x = x.reshape(b,n,self.w,self.low_dim)
        
        bias = bias.reshape(b,n,-1,self.low_dim)

        x_replace = self.random_replace(x = x_recon.detach().reshape(-1, self.w, self.low_dim),
                                        x_code =  x.reshape(-1, self.w, self.low_dim),
                                        min_replace=1, max_replace=self.w
                                       )[0].reshape(b, n, self.w, self.low_dim)

        noise_x = self.noise_x

        noise_w = self.noise_w.sample((b, n, self.w ,1)).to(x.device) if self.noise_pattern else None


        
        
        if self.is_dou: recon_continue_up = self.decoder_up(x + torch.randn_like(x)*(noise_w if self.noise_pattern else noise_x) if self.training else x)
        
        
        recon_continue_down =             self.decoder_down(x + torch.randn_like(x)*(noise_w if self.noise_pattern else noise_x) if self.training else x)
        
        
        if self.is_dou: recon_continue_bias_up = self.decoder_up(x + bias if self.training else x)
        
        
        recon_continue_bias_down =             self.decoder_down(x + bias if self.training else x)
        

        if self.training:

            point = 0

            if self.vql.decay != 1:
                

                x = torch.cat([x[:,:8], 
                               x_recon[:,8:] + torch.randn_like(x_recon[:,8:])*self.option_noise ], dim = 1)
                

            elif self.only_option_noise and self.vql.decay == 1:
                
                x = torch.cat([x[:, :8], x[:, 8:] + torch.randn_like(x[:, 8:])*self.reasoning_noise], dim = 1)

   
            else:
                x = x + torch.randn_like(x)*self.reasoning_noise
               

        K = self.num_aux_candidates
        
        if K > 0 and self.is_gumbel_vql == False:
            
            point_x = 0 if self.vql.decay == 1 else int(K/2)
            extra_candidates = torch.empty(b, 0, self.w, self.low_dim, device = x.device) if point_x == 0 else self.sample_from_codebook_topk(b, point_x, strategy = 'hard')  

            extra_candidates = torch.cat([extra_candidates, 
                                          (x if self.vql.decay == 1 else x_recon)[:, :8, :].reshape(-1, self.w, self.low_dim)[torch.randint(0, b*8, (b*(K - point_x), ))].reshape(b, K - point_x, self.w, self.low_dim)],
                                          
                                         dim = 1)
                                         
        elif K > 0 and self.is_gumbel_vql == True:
            point_x = max(K - 8, 0)
            extra_candidates = self.sample_from_codebook(b, point_x)  # [b, K, 16, dim]_topk
            extra_candidates = torch.cat([extra_candidates, 
                                          (x if self.vql.decay == 1 else x_recon)[:, 8:, :][:, :min(K, 8)]],
                                         dim = 1)

        
        else:
            extra_candidates = torch.empty(b, 0, 16, self.low_dim, device=x.device)
            
        
        noise_y = self.noise_y #max(noise_x - 1, 1)
        
        if self.is_dou:

            recon_up = torch.empty_like(recon_continue_up)

            recon_replace_up = torch.empty_like(recon_continue_bias_up)

        recon_down = torch.empty_like(recon_continue_down)

        recon_replace_down = torch.empty_like(recon_continue_bias_down)
        
        if self.is_dou and self.s_x != 0: recon_up =   self.decoder_up(x_recon.detach() + torch.randn_like(x_recon)*(noise_v if self.noise_pattern else noise_y) if self.training else x_recon)
        
        
        if self.s_x != 0:                 recon_down = self.decoder_down(x_recon.detach() + torch.randn_like(x_recon)*(noise_v if self.noise_pattern else noise_y) if self.training else x_recon)


        if self.is_dou and self.s_x != 0: recon_replace_up =   self.decoder_up(x_replace + torch.randn_like(x_replace)*(noise_v if self.noise_pattern else noise_y) if self.training else x_recon)
        
        
        if self.s_x != 0:                 recon_replace_down = self.decoder_down(x_replace + torch.randn_like(x_replace)*(noise_v if self.noise_pattern else noise_y) if self.training else x_recon)

        
        x = torch.cat([x, extra_candidates + (torch.randn_like(extra_candidates)*self.reasoning_noise if self.training else 0)], dim = 1)



        if self.j_mean == False: x = x + self.j_position[None, None]
        
 
        x = self.recat(x)
        
        x = self.g_function(x)# + torch.randn_like(x) if self.training else x) #

        qkv = self.graph_clip(self.recombine(x))
        
        qkv = qkv.chunk(self.num_rule, dim = -2)


        
        if self.j_mean:

        
            out =  map(lambda t: self.tajador(self.forward_cross_attention(t.squeeze(-2))).reshape(b, -1, self.w).mean(dim = -1), qkv)
            
        else:
            
            out =  map(lambda t: self.tajador(self.forward_cross_attention(t.squeeze(-2))).reshape(b, -1, self.w), qkv)
        
        y = 0 
        
        if self.is_dou:

            return *list(map(lambda t: t.mean(dim = 1).squeeze(-2), qkv)), x.reshape(-1, self.low_dim), \
                self.agg_out(out), y, bias.reshape(-1, self.w, self.low_dim), state, \
                     recon_up, recon_down,\
                             recon_continue_up, recon_continue_down,\
                                 recon_continue_bias_up, recon_continue_bias_down,\
                                     recon_replace_up, recon_replace_down, vq_loss, norm
                                     
        else:
            
            return *list(map(lambda t: t.mean(dim = 1).squeeze(-2), qkv)), x.reshape(-1, self.low_dim), \
                self.agg_out(out), y, bias.reshape(-1, self.w, self.low_dim), state, \
                      recon_down,\
                              recon_continue_down,\
                                  recon_continue_bias_down,\
                                      recon_replace_down, vq_loss, norm
                                 
        
    #"""
    def agg_out(self, out, select = None, agg='mean'):
        """
        对 map/iterable 结果按索引选择后聚合。
        select: None -> 默认全部 rule (range(num_rule))
                int  -> 单取该索引
                list/tuple -> 多索引
                slice    -> 切片
        agg: 'mean' | 'sum'
        """
        out_list = list(out)
        sel = [i for i in range(self.num_rule)] if select is None else select
    
        if isinstance(sel, int):
            return out_list[sel]
    
        # 统一转成列表
        if isinstance(sel, slice):
            selected = out_list[sel]
        elif isinstance(sel, (list, tuple)):
            selected = [out_list[i] for i in sel]
        else:
            raise TypeError(f"select must be int/list/tuple/slice, got {type(sel)}")
    
        if len(selected) == 0:
            raise ValueError("selected rule outputs is empty")
    
        if agg == 'sum':
            return sum(selected)
        elif agg == 'mean':
            return sum(selected) / len(selected)
        else:
            raise ValueError(f"agg must be 'mean' or 'sum', got {agg}")

    def my_cov(self, z):
        n = z.shape[0]
        d = z.shape[1]
        if n <= d :
            return 0
        # z = z.reshape(-1, self.laten_code)
        z_ = z.mean(dim = 0,keepdim = True)
        C_z = (torch.matmul((z - z_)[:,:,None], (z - z_)[:,None,:]).sum(dim = 0))/(n - 1)
        c_z = ((C_z**2)*(1 - torch.eye(d).to(z.device))).sum()/d
        return c_z
    
        
    def loss_function_ce(self, x, idx):
        # x: [b, C, w]，idx: [b]
        if x.dim() == 2:
            # 兼容旧调用：[b, C]
            return self.dio_loss(x, idx, delta=self.dio_delta), \
                   (x.argmax(dim=-1) == idx).float().sum()
    
        b, C, w = x.shape
    
        # 关键：不要对 w 做 mean；把 w 并进 batch 维
        logits = x.permute(0, 2, 1).reshape(b * w, C)      # [b*w, C]
        target = idx[:, None].expand(b, w).reshape(b * w)  # [b*w]
    
        loss = self.dio_loss(logits, target, delta=self.dio_delta)
    
        pred = logits.argmax(dim=-1)
        right = (pred == target).float().sum()/w  # 16 个 patch 的平均 choice accuracy
        # 如果你后面要 count，就改成 .sum()
    
        return loss, right
    
    def loss_function_sl(self, *out, target):
        


        
        keep_rule = (target != 7775)
        
        graph = out[0].mean(dim = 1, keepdim = True)# b 9 d
        
        # print(graph.shape)

        txt = out[1]# b t 7775 d
        # print(txt.shape)

        
        loss_1 = 0
    
        right = torch.zeros(1).sum().to(graph.device)
        
        if keep_rule.float().sum().item() != 0:
            

            r = F.cosine_similarity(graph[keep_rule,:,None, None], txt[:,None,:], dim = -1).mean(dim = -2) #b 5, t, 7775
            
            
            loss_1 += F.cross_entropy(r.reshape(-1, 7775)/ 1e-6, target[keep_rule, None].expand(-1, r.shape[1]).reshape(-1))

            right = (r.argmax(dim = -1).reshape(-1) == target[keep_rule, None].expand(-1, r.shape[1]).reshape(-1)).float().sum()/r.shape[1]

            """"""

        
        return loss_1, right, right/keep_rule.sum()
    
    
    def loss_function(self, *out, target_shape, target_line, idx):

        
        if self.is_dou:
            x_shape, x_line, _, _, z, x, y, bias, state, \
                 recon_up, recon_down, \
                         recon_continue_up, recon_continue_down, \
                             recon_continue_bias_up, recon_continue_bias_down, \
                                 recon_replace_up, recon_replace_down, vq_loss, norm = out
                            
        else:
            x_shape, x_line, _, _, z, x, y, bias, state, \
                 recon_down, \
                     recon_continue_down, \
                             recon_continue_bias_down, \
                                recon_replace_down, vq_loss, norm = out
            
                       

        

      

        if self.is_dou:
            
            loss_stright = (F.mse_loss(recon_up + recon_down - .5, state) if self.s_x !=0 else 0) + \
                    F.mse_loss(recon_continue_up + recon_continue_down - .5, state) + \
                        F.mse_loss(recon_continue_bias_up + recon_continue_bias_down - .5, state) + \
                            (F.mse_loss(recon_replace_up + recon_replace_down - .5, state) if self.s_x !=0 else 0)
            
    
            loss =  (F.mse_loss(recon_up, state) + F.mse_loss(recon_down, state)  if self.s_x !=0 else 0) +\
                        F.mse_loss(recon_continue_up, state) + F.mse_loss(recon_continue_down, state) +\
                            F.mse_loss(recon_continue_bias_up, state) + F.mse_loss(recon_continue_bias_down, state) +\
                                (F.mse_loss(recon_replace_up, state) + F.mse_loss(recon_replace_down, state) if self.s_x !=0 else 0)
            
        else:
            s_x = self.s_x
            s_y = 1
            
            loss_stright = (s_x*F.mse_loss(recon_down, state) if self.s_x !=0 else 0) +\
                        1*F.mse_loss(recon_continue_down, state) + s_y*F.mse_loss(recon_continue_bias_down, state) + \
                            (s_x*F.mse_loss(recon_replace_down, state)  if self.s_x !=0 else 0)
            
    
            loss =  0
            


        
        loss_3, right = self.loss_function_ce(x, idx)

        loss_4 = self.my_cov(z) 
        
        samlpes = torch.randn_like(bias)
        
        loss_5 = F.mse_loss(bias,  samlpes)

        


        norm_mean = norm.detach().mean()

        if self.print_norm: print(norm_mean.data)

        loss_norm = F.mse_loss(norm, torch.ones_like(norm)*self.norm_lower_bound) if self.toxic else 0

        return         self.gen_extent*(loss + 1000*loss_stright)\
                       + self.reasoning_extent*loss_3 + 1*loss_4 + self.bias_extent*loss_5 \
                       + self.vq_extent*torch.relu(vq_loss - self.vq_lower_bound) + self.vq_extent_2*torch.relu(vq_loss - self.vq_lower_bound_2) \
                       + (self.norm_extent*loss_norm if self.toxic else 0),\
                       loss_stright, \
                       vq_loss, \
                       right


################################################################################################################################################################################################
        
    def recon_all(self, state, lambd = 0):
        
        b, n, h, w = state.shape
 
        x = self.vit(state.view(b*n, 1, h, w))
        
        x, bias = x.chunk(2, dim = -1)
        
        x, _, _ = self.vql(x)
        
        x = x.reshape(b, n, self.w, self.low_dim)
        
        bias = bias.reshape(b, n, self.w, self.low_dim)
        
        x = x + torch.randn_like(x)*lambd + bias
   
        x_recon = self.decoder_up(x + bias) + self.decoder_down(x + bias) - 0.5

    
        return x_recon.reshape(b, n, h, w)
    
    
    def recon_randn_all(self, state, lambd = 1):
        
        b, n, h, w = state.shape
 
        x = self.vit(state.view(b*n, 1, h, w))
        
        x, bias = x.chunk(2, dim = -1)
        
        x, _, _ = self.vql(x)
        
        x = x.reshape(b, n, self.w, self.low_dim)
        
        bias = bias.reshape(b, n, self.w, self.low_dim)
        
        x = x + torch.randn_like(x)*lambd
        
        x_recon = self.decoder_up(x) + self.decoder_down(x) - 0.5

    
        return x_recon.reshape(b, n, h, w)



    def gumbel_nll_loss(self, logits, target, temperature=1.0, reduction='none'):
        """
        Gumbel-Softmax + NLLLoss
        
        Args:
            logits: [B, C] 原始 logits
            target: [B] 类别索引
            temperature: Gumbel 温度
        
        Returns:
            loss: 标量
        """
        # Gumbel 噪声
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits).clamp(1e-10, 1 - 1e-10)))
        
        # Gumbel-Softmax 对数概率
        log_probs = F.log_softmax((logits + gumbel_noise) / temperature, dim=1)
        
        # NLLLoss
        return F.nll_loss(log_probs, target, reduction= reduction)





    def dio_loss(self, logits, target, delta= None, gumbeling = False ):
        """
        纯 CE 思路实现 DIO 损失
        公式: CE + log( (1 - P_a + delta * P_a) / delta )
        logits : (B, 8)   9-16 号候选 logits
        target : (B,)     0-7 内的 gt 偏移
        delta  : 正例系数倒数 = 1/delta
        """
       
        if delta == None: delta = self.dio_delta

        if gumbeling:
            ce = self.gumbel_nll_loss(logits, target, reduction='none') 
            p = F.gumbel_softmax(logits, dim=-1)
            
        else:
            ce = F.cross_entropy(logits, target, reduction='none') 
            p = F.softmax(logits, dim=-1)
    
      
        
        if delta<1:
            p_alpha = p.gather(1, target.unsqueeze(1)).squeeze(1)  # (B,)
        

            corr = torch.log((1 - p_alpha + delta * p_alpha) / delta)
            
        else:
            corr = 0
    
        return (ce + corr).mean()
    
    def recon_randn_replace(self, state, lambd = 0):
        
        b, n, h, w = state.shape

        # print(state.shape)
 
        x = self.vit(state.view(b*n, 1, h, w))
        
        x, bias = x.chunk(2, dim = -1)
        
        x, _, _ = self.vql(x)
        
        x, _ = self.random_replace(x = x, max_replace=self.max_replace)
        
        # print(x.shape)
        x = x.reshape(b, n, self.w, self.low_dim)
        
        bias = bias.reshape(b, n, self.w, self.low_dim)
        

        x = x + torch.randn_like(x)*lambd
        
        x_recon = self.decoder_up(x) + self.decoder_down(x) - 0.5

    
        return x_recon.reshape(b, n, h, w)
        
    def random_replace(self, x, x_code = None, min_replace=1, max_replace=9):
        b, s, d = x.shape
        assert s >= max_replace
        device = x.device
        
        if x_code is None:
            x_code = self.sample_from_codebook_topk(b, 1).reshape(b, s, d).detach()
        

        count = torch.randint(min_replace, max_replace + 1, (b,), device=device)
        

        ranks = torch.rand(b, s, device=device).argsort(dim=1).argsort(dim=1)
        

        mask = (ranks < count.unsqueeze(1)).unsqueeze(-1).expand(-1, -1, d)
        

        
        return torch.where(mask, x_code, x), count

    def add_noise_to_patch(self, x, min_noise=1, max_noise=8, noise_scale=5, zero_padding_p = 0.3):
        b, s, d = x.shape
        device = x.device
        

        count = torch.randint(min_noise, max_noise + 1, (b,), device=device)
        
        zero_padding = (torch.rand(b, device=device) > zero_padding_p).float()
        
        count = count * zero_padding
        

        rand = torch.rand(b, s, device=device)
        ranks = rand.argsort(dim=1).argsort(dim=1)  # 排名
        

        mask = ranks < count.unsqueeze(1)
        
       
        noise = torch.randn_like(x) * noise_scale
        return torch.where(mask.unsqueeze(-1).expand(-1, -1, d), (x + noise).data, x)
    
    
    def sample_from_codebook(self, b, k=None):

        with torch.no_grad():
            embed = self.vql.embed
            is_pos = embed.dim() == 4
    
            if is_pos:
                P, H, D, N = embed.shape
            else:
                H, D, N = embed.shape
    
            if k is None:
                k = self.num_aux_candidates
            w = self.w
    
            if is_pos:
                assert P == w, f"codebook positions {P} != w {w}"
                # 按 (pos, head, code) 打平: [P, H, D, N] -> [P*H*N, D]
                embed_flat = embed.permute(0, 1, 3, 2).reshape(P * H * N, D)
            else:
                # 按 (head, code) 打平: [H, D, N] -> [H*N, D]
                embed_flat = embed.permute(0, 2, 1).reshape(H * N, D)
    
            rand_idx = torch.randint(0, N, (b, k, w, H), device=embed.device)
    
            head_offsets = torch.arange(H, device=embed.device).view(1, 1, 1, H) * N
    
            if is_pos:
                pos_offsets = torch.arange(P, device=embed.device).view(1, 1, P, 1) * (H * N)
                global_idx = rand_idx + head_offsets + pos_offsets   # [b, k, w, H]
            else:
                global_idx = rand_idx + head_offsets                 # [b, k, w, H]
    
            samples = F.embedding(global_idx, embed_flat)            # [b, k, w, H, D]
            samples = samples.reshape(b, k, w, H * D)
    
        return samples.detach()
    
    
    def sample_from_codebook_topk(self, b, k=None, top_k=None, strategy='hard',
                                  temperature=1.0, perturb_scale=.1):

        with torch.no_grad():
            embed = self.vql.embed
            is_pos = embed.dim() == 4
    
            if is_pos:
                P, H, D, N = embed.shape
            else:
                H, D, N = embed.shape
    
            if k is None:
                k = self.num_aux_candidates
            w = self.w
    
            if is_pos:
                assert P == w, f"codebook positions {P} != w {w}"
    
            if top_k is None:
                top_k = int(N * 0.55)
    
            # 聚类表未就绪时回退到均匀随机采样（内部已兼容两种版本）
            if not self.vql.map_sealed and self.vql.sample_count < 10000:
                print(f"[Warning] Cluster map not ready ({self.vql.sample_count}), fallback to random")
                return self.sample_from_codebook(b, k)
    
            # cluster_map: [H, w, N]，两版本一致
            cluster_map = self.vql.get_cluster_map(normalize=False)
    
            head_offsets = torch.arange(H, device=embed.device).view(1, 1, H) * N
    
            samples_list = []
    
            for pos in range(w):
                # 该位置在所有头上的使用频次 [H, N]
                pos_usage = cluster_map[:, pos, :]
    
                if strategy == 'hard':
                    # === 硬Top-K：只从高频码本中均匀选择 ===
                    _, topk_indices = torch.topk(pos_usage, top_k, dim=-1)      # [H, top_k]
                    rand_in_topk = torch.randint(0, top_k, (b, k, H), device=embed.device)
                    selected_indices = torch.gather(
                        topk_indices.unsqueeze(0).unsqueeze(1).expand(b, k, -1, -1),
                        -1,
                        rand_in_topk.unsqueeze(-1)
                    ).squeeze(-1)                                            # [b, k, H]
    
                elif strategy == 'soft':
                    # === 软Top-K：按频率加权多项式采样 ===
                    values, indices = torch.topk(pos_usage, top_k, dim=-1)     # [H, top_k]
                    probs = F.softmax(values / temperature, dim=-1)            # [H, top_k]
                    selected_in_topk = torch.multinomial(
                        probs.unsqueeze(0).unsqueeze(1).expand(b, k, -1, -1).reshape(-1, top_k),
                        num_samples=1,
                        replacement=True
                    ).reshape(b, k, H)                                       # top_k 内的索引
                    selected_indices = torch.gather(
                        indices.unsqueeze(0).unsqueeze(1).expand(b, k, -1, -1),
                        -1,
                        selected_in_topk.unsqueeze(-1)
                    ).squeeze(-1)                                            # [b, k, H]
    
                elif strategy == 'mix':
                    # === 混合策略：80% 高频 + 20% 随机探索 ===
                    mix_mask = torch.rand(b, k, H, device=embed.device) < 0.8
    
                    _, topk_indices = torch.topk(pos_usage, top_k, dim=-1)
                    rand_in_topk = torch.randint(0, top_k, (b, k, H), device=embed.device)
                    hard_samples = torch.gather(
                        topk_indices.unsqueeze(0).unsqueeze(1).expand(b, k, -1, -1),
                        -1,
                        rand_in_topk.unsqueeze(-1)
                    ).squeeze(-1)
    
                    random_samples = torch.randint(0, N, (b, k, H), device=embed.device)
    
                    selected_indices = torch.where(mix_mask, hard_samples, random_samples)
    
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")
    
                global_idx = selected_indices + head_offsets               # [b, k, H]
    
                # ✅ 关键分支：position 版按位置取自己的码本，老版用全局码本
                if is_pos:
                    pos_embed_flat = embed[pos].permute(0, 2, 1).reshape(H * N, D)   # embed[pos]: [H, D, N]
                else:
                    pos_embed_flat = embed.permute(0, 2, 1).reshape(H * N, D)
    
                pos_samples = F.embedding(global_idx.reshape(-1), pos_embed_flat).reshape(b, k, H, D)
                samples_list.append(pos_samples)
    
            # 合并所有位置: [b, k, w, H, D] -> [b, k, w, H*D]
            samples = torch.stack(samples_list, dim=2).reshape(b, k, w, H * D)
    
            return samples.detach()
    

       

        
def transpose(x):
    return x.transpose(-2, -1).contiguous()
    
def mul_dot(a, b):
    
    assert a.dim() == b.dim() and a.dim() == 3 and b.shape[1] == 7776  and a.shape[1] == 1, 'error'
    
    # a@transpose(b)
    
    # print(a.shape,b.shape, (a@transpose(b)).shape)
    return (a@transpose(b)).squeeze(-1)
    
    
def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]          
        
        
def reasoning(*args):
  return raven_clip()




 
if __name__ == '__main__':
    #from torchsummary import summary
    
    num_item = 1
    x = torch.randn(num_item,16,80,80)
    y = torch.randint(1,(num_item,7776,16)).long()
    target = torch.randint(7776,(num_item,)).long()
    label = torch.randint(8,(num_item,)).long()
    
    model = raven_clip()

    out = model(x)
    
    l, right_shape, right_line, right = model.loss_function(*out, target_shape = target, target_line = target, idx = label)
    l.backward()

    print(model)
    from torchinfo import summary

    summary(model.cuda(), input_size=(1, 16, 80, 80),
        col_names=["input_size", "output_size", "num_params", 
                    "kernel_size", "mult_adds", "trainable"], device='cpu')
    #accuracy = model.choose_accuracy(*out, idx = label)
    from torch.utils.benchmark import Timer
    stmt = "model(input_tensor)"
    setup = "model.eval(); torch.cuda.synchronize()"
    model = raven_clip(num_aux_candidates = 0)
    device = torch.device("cuda")
    #device = torch.device("cpu")
    timer = Timer(
        stmt=stmt,
        setup=setup,
        globals={"model": model.cuda(), "input_tensor": x.cuda()},
        num_threads=4,
        label="Latency",
        sub_label="batch=1"
    )
    
    result = timer.blocked_autorange(min_run_time=20)  # 至少跑10秒
    print(f"{result.median*1000:.3f} ms ± {result.iqr*1000:.3f} ms")

