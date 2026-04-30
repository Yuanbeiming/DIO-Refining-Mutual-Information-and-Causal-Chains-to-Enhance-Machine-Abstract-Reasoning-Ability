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

def round_tensor(x, decimals=2):
    factor = 10 ** decimals
    return torch.floor(x * factor + 0.5) / factor   

class raven_clip(nn.Module):
    def __init__(self, *args, num_aux_candidates = 0):
        super(raven_clip,self).__init__()

        #self.num_embeddings = 8192
        

        

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

        


        self.name = 'DIO_DIEGO_validition'
            
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
        
        self.num_aux_candidates = num_aux_candidates 


        
        num_candidates = num_aux_candidates + 8
        
        m = (2+num_candidates)*2


        self.temperature = temperature

        
        self.num_rule = 4
         
   
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
                            
                            Rearrange('(b m s n) d -> b m (s n d)', s = self.w, m = num_candidates, n = 1),
                            Mean(dim = -1))
        
        
        txt_num_cls = 1
        self.txt_clip = nn.Sequential(Rearrange('b n s -> (b n) s', s = 10, n = txt_size),
                            txt_mask_transformer(dict_size = 14, words = 10, dim = self.low_dim, depth = 3, 
                                                 heads = num_head, dim_head = int(self.low_dim/num_head), mlp_dim = self.low_dim*2, dropout = 0.1,is_pgm = True, 
                                                 num_cls=min(1, txt_num_cls)),
                            Mean(dim = 1) if txt_num_cls == 0 else take_cls(txt_num_cls),
                            # take_cls(),
                            Rearrange('(b n) d -> b n d', n = txt_size)) #b,336,d

        
        
        #self._add_spectral_norm()
        self.replace_x = 0
        self.max_replace = int(self.w/2)
        self.noise_x = 1

################################################################################################################################################################################################
        self.pretrain = True

        if self.pretrain:
            
            

            pretrained_params = torch.load('C:/Users/DELL/Desktop/DIO_code/generate model/model_DIO_DIEGO_validation.pt', map_location = 'cpu')


           
            for name, param in self.named_parameters():
                if name in pretrained_params:#
                    
                    if name[:3] == 'vit':
                             param.requires_grad = False   
                else:
                    print(f"Warning: Parameter '{name}' not found in pretrained dict.")
                        
                    
             
            for name, buffer in self.named_buffers():
                if name in pretrained_params:
                    if name[:3] != 'vql':
                         buffer.data.copy_(pretrained_params[name].data)

                else:
                    print(f"Warning: Buffer '{name}' not found in pretrained dict.")

        print('num_aux_candidates:', num_aux_candidates)
        print('noise_x is:', self.noise_x)


    
        

    
    def _add_spectral_norm(self):
        
        
        exclude_names = {'vit', 'txt_clip'}
        
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
        
        x = x.view(b*n, 1, h, w)
        
        x = self.vit(x)
        
        x, _ = x.chunk(2, dim = -1)
        
        x = x.reshape(b,n,-1,self.low_dim)
        
        extra_candidates = torch.empty(b, 0, 16, self.low_dim, device=x.device)
        
        x = torch.cat([x, extra_candidates], dim = 1)

        x = self.recat(x)
        
        x = self.g_function(x + torch.randn_like(x)*self.noise_x if self.training else x)

        qkv = self.graph_clip(self.recombine(x))
        
        qkv = qkv.chunk(self.num_rule, dim = -2)#b s n c d
        

        
        out =  map(lambda t: self.tajador(self.forward_cross_attention(t.squeeze(-2))), qkv)
        
        

        y = self.txt_clip(self.txt_data.to(x.device))
        

                
        return *list(map(lambda t: t.mean(dim = 1).squeeze(), qkv)), x.reshape(-1, self.low_dim), \
            self.agg_out(out), y
        
    #"""
    def agg_out(self, out, select=[1,2,3,4], agg='mean'):
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
        
        return self.dio_loss(x, idx), (x.argmax(dim = -1) == idx).float().sum() if self.training else (x[:, :8].argmax(dim = -1) == idx).float().sum()
        """
        loss = self.dio_loss(x, idx, reduction='none' )
        
        
        is_wrong = (x.argmax(dim = -1) != idx).float()*1 + 1


        return (loss*is_wrong).mean(),\
            (x.argmax(dim = -1) == idx).float().sum() if self.training else (x[:, :8].argmax(dim = -1) == idx).float().sum()
        """
    
    def loss_function_sl(self, *out, target):
        


        
        keep_rule = (target != 7775)
        
        graph = out[0].mean(dim = 1, keepdim = True)# b 9 d
        
        

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


        # _, _, x_shape, x_line, z, x, y, bias, state, recon_up, recon_down, recon_bias_up, recon_bias_down = out
        
        x_shape, x_line, _, _,  z, x, y = out
                       
        idx_ = F.one_hot(idx, 8)[:,:,None,None]
        
        x_shape = (x_shape[:,:8]*idx_).sum(dim = 1)
        
        x_line = (x_line[:,:8]*idx_).sum(dim = 1)
        
 
        y = y.unsqueeze(1)
        
        # self.replace_x = 1 - (self.max_replace/self.w)
       
        # loss = F.mse_loss(recon_up, state) + F.mse_loss(recon_bias_up, state) +\
        #     F.mse_loss(recon_down, state) + F.mse_loss(recon_bias_down, state)
        
        
        loss_1, right_shape, rr_shape = self.loss_function_sl(x_shape, y, target = target_shape)
        
        loss_2, right_line, rr_line = self.loss_function_sl(x_line, y, target = target_line)
        
        
        
        loss_3, right = self.loss_function_ce(x, idx)

        loss_4 = self.my_cov(z)
        

       
        return 10*(loss_1 + loss_2) + 50*loss_3 + 0.1*loss_4, rr_shape,  rr_line, right
        # null_embdv

################################################################################################################################################################################################
        
    


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





    def dio_loss(self, logits, target, delta= 1, gumbeling = False, reduction='mean' ):
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
            
        if reduction == 'mean':
            return (ce + corr).mean()#, (logits.argmax(dim = -1) == target).float().sum()
        
        elif reduction == 'sum':
            return (ce + corr).sum()
        
        else:
            return ce + corr
        
        
        
    
       

        
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
    
    x = torch.randn(2,16,80,80)
    y = torch.randint(1,(2,7776,16)).long()
    target = torch.randint(7776,(2,)).long()
    label = torch.randint(8,(2,)).long()
    
    model = raven_clip()
    
    model.eval()
    torch.save(model.state_dict() , './model_DIO_DIEGO_validition_.pt')
    
#%%   
    out = model(x)
    #%%
    l, right_shape, right_line, right = model.loss_function(*out, target_shape = target, target_line = target, idx = label)
    l.backward()

    print(model)
    from torchinfo import summary
    #model = raven_clip().to('cuda' if torch.cuda.is_available() else 'cpu')
    #summary(model, (16, 80, 80), device='cpu')
    summary(model, input_size=(2, 16, 80, 80),
        col_names=["input_size", "output_size", "num_params", 
                    "kernel_size", "mult_adds", "trainable"], device='cpu')
    #accuracy = model.choose_accuracy(*out, idx = label)
    

