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
    def __init__(self, c, h ,w):
        super().__init__()
        
        self.linear = nn.Linear(c*h*w, c*h*w)
        
    def forward(self, x):
        
        b, c, h, w = x.shape
        
        x = x + self.linear(x.reshape(b, -1)).reshape(b, c, h, w)


        return x



class raven_clip(nn.Module):
    def __init__(self, *args, num_aux_candidates =  4):
        super(raven_clip,self).__init__()

        #self.num_embeddings = 8192
        self.num_embeddings = 2**14

        

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

        vql_heads = 2


        self.name = 'DIO_WORLD_sn_embdv'+str(self.num_embeddings)+ '_heads_' +str(vql_heads)
            
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
        
        self.num_aux_candidates = num_aux_candidates #int((16**vql_heads)/self.w)


        
        num_candidates = num_aux_candidates + 8
        
        m = (2+num_candidates)*2


        self.temperature = temperature

        
        self.num_rule = 2

        
            
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
        
        #self.discrimnator = SinkhornDistance(eps=0.1, max_iter = 100, reduction='mean')

        num_decoder_depth = num_depth*2 + 2
        
        self.decoder_up = nn.Sequential(Rearrange('b n s d -> (b n) s d',  s = self.w),
                                      # Mean(dim = -2, keepdim = True),
                                          ViT_reverse(#words = self.w, 
                                                               
                                                                image_size = 80,  
                                                               
                                                                patch_size = 20,
                                                               
                                                                channels = 1,
                                                               
                                                                dim = self.low_dim, 
                                                               
                                                                depth = num_decoder_depth,
                                                                
                                                                heads = num_head,
                                                                
                                                                mlp_dim = self.low_dim,
                                                                
                                                                dim_head = int(self.low_dim/num_head),
                                                                
                                          ), Sigmoid_up()
                                          
                                          
                                          )
        
        self.decoder_down = nn.Sequential(Rearrange('b n s d -> (b n) s d',  s = self.w),
                                      # Mean(dim = -2, keepdim = True),
                                          ViT_reverse(#words = self.w, 
                                                               
                                                                image_size = 80,  
                                                               
                                                                patch_size = 20,
                                                               
                                                                channels = 1,
                                                               
                                                                dim = self.low_dim, 
                                                               
                                                                depth = num_decoder_depth,
                                                                
                                                                heads = num_head,
                                                                
                                                                mlp_dim = self.low_dim,
                                                                
                                                                dim_head = int(self.low_dim/num_head),
                                                                
                                          ),Sigmoid_down())
        
        
        
        
        self.g_function = nn.Sequential(
            Rearrange('b s m d -> (b s m) d'),

            Bottleneck_judge(3*self.low_dim, int(self.low_dim*0.75), 3*self.low_dim),#10,10
            
            Bottleneck_judge(3*self.low_dim, int(self.low_dim*0.75), 3*self.low_dim),#10,10
            
            Bottleneck_judge(3*self.low_dim, int(self.low_dim*0.75), self.low_dim),#10,10
            
            Rearrange('(b s m) d -> b s m d', s = self.w, m = m),
            
            # Recombine(),
            

            
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
                            
                            Rearrange('(b m s n) d -> b m (s n d)', s = self.w, m = num_candidates, n = 1),
                            Mean(dim = -1))

        
        self.vql = VectorQuantizerEMA_multi_head_revival(self.num_embeddings,
			                                        self.low_dim,
			                                        vql_heads,
			                                        self.beta,
			                                        0.995,
			                                        vq_loss_type = 'mse')
        self._add_spectral_norm()
        self.replace_x = 0
        self.max_replace = 9
################################################################################################################################################################################################
        if self.vql.decay < 1:
            print('val_decay:', self.vql.decay)
        else:
            print('*'*100)
            print('Worning! embed is fixed')
        print('num_aux_candidates:', num_aux_candidates)
        

    def _add_spectral_norm(self):
        
        
        exclude_names = {'vit', 'decoder_up', 'decoder_down'}
        
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

    def sample_from_codebook(self, b, k = None):
        
        with torch.no_grad():
            embed = self.vql.embed  # [H, D, N]
            H, D, N = embed.shape
            if k is None:
                k = self.num_aux_candidates  # 采样数K
            w = self.w  # patch数量(16)

            rand_idx = torch.randint(
                0, N, 
                (b, k, w, H), 
                device=embed.device
            )
            
            embed_flat = embed.permute(0, 2, 1).reshape(H * N, D)
            
            head_offsets = torch.arange(H, device=embed.device).view(1, 1, 1, H) * N
            global_idx = rand_idx + head_offsets  # [b, k, w, H]
            
            samples = F.embedding(global_idx, embed_flat)
            
            samples = samples.reshape(b, k, w, H * D)
            
        return samples.detach()  # [b, k, w, low_dim]
       
    
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

        x_recon, vq_loss, _ = self.vql(x)
        
        x_recon_replaced, self.replace_x = self.random_replace(x = x_recon.detach(), max_replace=self.max_replace)
        
        assert self.replace_x <= 1.0
    
        #x_recon, vq_loss = x, torch.zeros(1, device = x.device).sum()
        
        x_recon = x_recon.view(b, n, self.w, self.low_dim)

        x_recon_replaced = x_recon_replaced.view(b, n, self.w, self.low_dim)

        if self.training:
            point = int(b/2)
    
            x = torch.cat([x.reshape(b,n,-1,self.low_dim)[:point], x_recon[point:]], dim = 0)
        
        else:
            x = x.reshape(b,n,-1,self.low_dim)
        
        #x = x.reshape(b,n,-1,self.low_dim)


        bias = bias.reshape(b,n,-1,self.low_dim)
        
        
        K = self.num_aux_candidates
        
        if K > 0:
            
            point_x = 0 #int(K/2)
            extra_candidates =                   self.sample_from_codebook(b)[:, :point_x]  # [b, K, 16, dim]
            extra_candidates = torch.cat([extra_candidates, 
                                          x_recon[:, :8, :].reshape(-1, self.low_dim)[torch.randint(0, b*8*self.w, (b*(K - point_x)*self.w, ))].reshape(b, K - point_x, self.w, self.low_dim)],
                                         dim = 1)
            """
            #extra_candidates = self.sample_from_codebook(b)  # [b, K, 16, dim]
            extra_candidates = x[:, :8, :].reshape(-1, self.low_dim)[torch.randint(0, b*8*self.w, (b*K*self.w, ))].reshape(b, K, self.w, self.low_dim)
            """
            
            # extra_bias = torch.randn_like(extra_candidates)
        else:
            extra_candidates = torch.empty(b, 0, 16, self.low_dim, device=x.device)
        
        
        recon_bias_up = self.decoder_up(x_recon + bias)
        
        recon_bias_down = self.decoder_down(x_recon + bias)

        
        recon_up = self.decoder_up(x_recon + torch.randn_like(x_recon) if self.training else x_recon)
        
        
        recon_down = self.decoder_down(x_recon + torch.randn_like(x_recon) if self.training else x_recon)
        
        
        recon_replace_up = self.decoder_up(x_recon_replaced + torch.randn_like(x_recon) if self.training else x_recon)
        
        
        recon_replace_down = self.decoder_down(x_recon_replaced + torch.randn_like(x_recon) if self.training else x_recon)
        

        x = torch.cat([x, extra_candidates], dim = 1)
 
        x = self.recat(x)
        
        x = self.g_function(x + torch.randn_like(x) if self.training else x)

        qkv = self.graph_clip(self.recombine(x))
        
        qkv = qkv.chunk(self.num_rule, dim = -2)
        

        
        out =  map(lambda t: self.tajador(self.forward_cross_attention(t.squeeze(-2))), qkv)
        
        
        # print(out.shape)
        y = torch.randn(1,1).to(x.device)#self.txt_clip(self.txt_data.to(x.device))

        return *list(map(lambda t: t.mean(dim = 1).squeeze(), qkv)), x.reshape(-1, self.low_dim), \
            sum(list(out)), y, bias.reshape(-1, self.w, self.low_dim), state, \
                recon_up, recon_down, recon_bias_up, recon_bias_down, recon_replace_up, recon_replace_down, vq_loss
        
    #"""

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
        


        return self.dio_loss(x, idx), (x.argmax(dim = -1) == idx).float().sum() if self.training else (x[:, :8].argmax(dim = -1) == idx).float().sum()
        
        
        #return self.dio_loss(x, idx)F.cross_entropy(x/1,idx)

    

    
    
    
    def loss_function(self, *out, target_shape, target_line, idx):
        
        # idx = None
        x_shape, x_line, z, x, y, bias, state, recon_up, recon_down, recon_bias_up, recon_bias_down, recon_replace_up, recon_replace_down, vq_loss = out
        
 
        y = y.unsqueeze(1)
        
        # self.replace_x = 1 - (self.max_replace/self.w)
       
        
        loss = F.mse_loss(recon_up, state) + F.mse_loss(recon_bias_up, state) +\
            F.mse_loss(recon_down, state) + F.mse_loss(recon_bias_down, state) +\
                self.replace_x*F.mse_loss(recon_replace_up, state) + self.replace_x*F.mse_loss(recon_replace_down, state)

        loss_stright = F.mse_loss(recon_up + recon_down - .5, state) + \
            F.mse_loss(recon_bias_up + recon_bias_down - .5, state) + \
                self.replace_x*F.mse_loss(recon_replace_up + recon_replace_down - .5, state)
        
        right_shape = torch.zeros(1).sum().to(x.device)
        
        loss_1 = 0

        right_line = torch.zeros(1).sum().to(x.device)

        loss_2 = 0
        """"""
        """
        
        loss_1, right_shape = self.loss_function_sl(x_shape, y, target = target_shape)
        
        loss_2, right_line = self.loss_function_sl(x_line, y, target = target_line)

        """
        
        loss_3, right = self.loss_function_ce(x, idx)

        loss_4 = self.my_cov(z)

        """

        if self.training:
            
            self.beta.step()

        """
        
        samlpes = torch.randn_like(bias)
        
    
        
        loss_5 = F.mse_loss(bias,  samlpes)


       
        #return 100*(loss + 100*loss_stright) + 10*loss_3 + 1*loss_4 + loss_5 + 5*torch.relu(vq_loss - 0.1) + 10*torch.relu(vq_loss - 0.64) + 100*torch.relu(vq_loss - 0.99), loss_stright,  vq_loss, right
        #2^14_embdv

        return 100*(loss + 100*loss_stright) + 10*loss_3 + 1*loss_4 + loss_5 + 10*torch.relu(vq_loss - 0.1) + 10*torch.relu(vq_loss - 0.64) + 100*torch.relu(vq_loss - 0.99), loss_stright,  vq_loss, right
        #2^14_embdv
        #return 100*(loss + 100*loss_stright) + 20*loss_3 + 1*loss_4 + loss_5 + 5*torch.relu(vq_loss - 0.1) + 10*torch.relu(vq_loss - 0.64) + 100*torch.relu(vq_loss - 0.99), loss_stright,  vq_loss, right
        #2048_embdv
        
        
        #return 50*(loss + 10*loss_stright) + 10*loss_3 + 1*loss_4 + loss_5 + vq_loss, loss_stright,  vq_loss, right
        # null_embdv

################################################################################################################################################################################################
        
    def recon_all(self, state, lambd = 0):
        
        b, n, h, w = state.shape

        # print(state.shape)
 
        x = self.vit(state.view(b*n, 1, h, w))
        
        x, bias = x.chunk(2, dim = -1)
        
        x, _, _ = self.vql(x)
        
        # print(x.shape)
        x = x.reshape(b, n, self.w, self.low_dim)
        
        bias = bias.reshape(b, n, self.w, self.low_dim)
        
        
        x = x + torch.randn_like(x)*lambd + bias
        
        
        
        x_recon = self.decoder_up(x + bias) + self.decoder_down(x + bias) - 0.5
        # x_recon = self.decoder(x)
    
        return x_recon.reshape(b, n, h, w)
    
    
    def recon_randn_all(self, state, lambd = 1):
        
        b, n, h, w = state.shape

        # print(state.shape)
 
        x = self.vit(state.view(b*n, 1, h, w))
        
        x, bias = x.chunk(2, dim = -1)
        
        x, _, _ = self.vql(x)
        
        # print(x.shape)
        x = x.reshape(b, n, self.w, self.low_dim)
        
        bias = bias.reshape(b, n, self.w, self.low_dim)
        
        # bias = torch.tanh(bias)
        # print(x.shape)
        x = x + torch.randn_like(x)*lambd
        x_recon = self.decoder_up(x) + self.decoder_down(x) - 0.5
        # x_recon = self.decoder(x)
    
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





    def dio_loss(self, logits, target, delta= 1, gumbeling = False ):
        """
        纯 CE 思路实现 DIO 损失
        公式: CE + log( (1 - P_a + delta * P_a) / delta )
        logits : (B, 8)   9-16 号候选 logits
        target : (B,)     0-7 内的 gt 偏移
        delta  : 正例系数倒数 = 1/delta
        """
        # 1. 交叉熵项（reduction='none' 保留每个样本的 loss）
        #ce = F.cross_entropy(logits, target, reduction='none')   # -log P_a
        
        if gumbeling:
            ce = self.gumbel_nll_loss(logits, target, reduction='none') 
            p = F.gumbel_softmax(logits, dim=-1)
            
        else:
            ce = F.cross_entropy(logits, target, reduction='none') 
            p = F.softmax(logits, dim=-1)
    
        # 2. 取 P_a
        #p = F.softmax(logits, dim=-1)            # (B, 8)
        
        if delta<1:
            p_alpha = p.gather(1, target.unsqueeze(1)).squeeze(1)  # (B,)
        
            # 3. 修正项
            corr = torch.log((1 - p_alpha + delta * p_alpha) / delta)
            
        else:
            corr = 0
    
        return (ce + corr).mean()#, (logits.argmax(dim = -1) == target).float().sum()
    
    
    def random_replace(self, x, x_code = None, min_replace=1, max_replace=9):
        b, s, d = x.shape
        assert s >= max_replace
        device = x.device
        
        if x_code is None:
            x_code = self.sample_from_codebook(b, 1).reshape(b, s, d).detach()
        
        # 每个batch独立随机替换个数
        count = torch.randint(min_replace, max_replace + 1, (b,), device=device)
        
        # print(count)
        
        # 两次argsort直接得排名，无需scatter
        ranks = torch.rand(b, s, device=device).argsort(dim=1).argsort(dim=1)
        
        # print(ranks)
        
        # 排名 < count 即被选中
        mask = (ranks < count.unsqueeze(1)).unsqueeze(-1).expand(-1, -1, d)
        
        # print(mask)
        
        return torch.where(mask, x_code, x), 1 - (count.sum()/(b*s))

        
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
    from torchinfo import summary
    x = torch.randn(2,16,80,80)
    y = torch.randint(1,(2,7776,16)).long()
    target = torch.randint(7776,(2,)).long()
    label = torch.randint(8,(2,)).long()
    
    model = raven_clip()
    
    model.eval()
    # params = torch.load('./model_Clip_raven_120000_distribute_nine_best.pt', map_location = 'cpu')
    # # model_dict =  model.state_dict()
    
    # # state_dict = {k:v for k,v in params.items() if k in model_dict.keys()}
    # for k,q in model.named_parameters():
    #     if k[:7] != 'tajador':
    #         print(k)
    #         q.data = params[k].data
            

    # 
    # model_dict.update(state_dict)
    # model.load_state_dict(model_dict)       
    # model.load_state_dict(torch.load('./model_Clip_raven_120000_distribute_nine_best.pt', map_location = 'cpu'))
    out = model(x)
    
    l, right_shape, right_line, right = model.loss_function(*out, target_shape = target, target_line = target, idx = label)
    l.backward()

    print(model)

    #model = raven_clip().to('cuda' if torch.cuda.is_available() else 'cpu')
    #summary(model, (16, 80, 80), device='cpu')
    summary(model, input_size=(2, 16, 80, 80),
        col_names=["input_size", "output_size", "num_params", 
                    "kernel_size", "mult_adds", "trainable"], device='cpu')
    #accuracy = model.choose_accuracy(*out, idx = label)
    

    
    
