# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 18:06:07 2026

@author: 28905
"""

# -*- coding: utf-8 -*-
"""
测试脚本：加载训练好的模型，在验证/测试集上评估精度
（由训练脚本改造而来，已删除优化器、学习率调度、训练循环、模型保存等训练相关代码）
"""

import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import random

import torch.backends.cudnn as cudnn
from tqdm import tqdm

import DIO_WORLD_GEN as model_vit
import make_pgm_data as make_data


def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


t = transforms.Resize((80, 80))

batch_size = 100

# ===================== 测试集 =====================
val_set = make_data.Raven_Data(train=False, val=True)
num_val = len(val_set)

val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                        num_workers=4, prefetch_factor=4, pin_memory=True)

print('test samples:', num_val)

# ===================== 设备 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

init_seeds(2048)

# ===================== 模型 =====================
model = model_vit.raven_clip()

# 模型命名规则与训练脚本保持一致，用于定位权重文件
# 注意：len_train_set 需要与训练时使用的训练集大小一致
len_train_set = 0  # TODO: 填训练时的训练集样本数；或直接在下面指定 checkpoint 路径

if model_vit.big == True:
    name = 'big_' + model.name + '_' + str(len_train_set) + '_' + make_data.aaa
    text_name = 'big_' + model.name + make_data.aaa
else:
    name = model.name + '_' + str(len_train_set) + '_' + make_data.aaa
    text_name = model.name + make_data.aaa
print(name)

# ===================== 加载权重 =====================
checkpoint = './model_' + name + '_best.pt'   # 也可以直接写死路径，如 './model_xxx_best.pt'
model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
print('model parameters loaded from:', checkpoint)

model = model.to(device)

# ===================== 测试 =====================
model.eval()  # 不启用 Batch Normalization 和 Dropout

accuracy_val = [0] * 5
drop_test_sample = [0] * 2

with torch.no_grad():
    with tqdm(total=len(val_loader)) as pbar:
        for x_test, label, label_in, idx in val_loader:

            x_test = t(x_test).float().to(device)

            label = label.long().to(device)
            label_in = label_in.long().to(device)
            idx = idx.long().to(device)

            drop_test_sample[0] += label.eq(7775).sum().item()
            drop_test_sample[1] += label_in.eq(7775).sum().item()

            x_test = x_test / 255.

            out_test = model(x_test)

            _, right, right_in, choose_right = model.loss_function(
                *out_test, target_shape=label, target_line=label_in, idx=idx)

            accuracy_val[0] += right.cpu().numpy()
            accuracy_val[1] += right_in.cpu().numpy()
            accuracy_val[2] += choose_right.cpu().numpy()

            pbar.set_postfix(
                accuracy_batch=(choose_right.sum() / (x_test.shape[0] * 2)).item())
            pbar.update(1)

# ===================== 统计结果 =====================
if num_val - drop_test_sample[0] == 0:
    accuracy_val[0] = 0
else:
    accuracy_val[0] /= (num_val - drop_test_sample[0])

if num_val - drop_test_sample[1] == 0:
    accuracy_val[1] = 0
else:
    accuracy_val[1] /= (num_val - drop_test_sample[1])

accuracy_val[2] /= num_val

result = "accuracy_val: shape:{:.4f}\t line:{:.4f}\t choose:{:.4f}\n".format(*accuracy_val[:3])

print(result)

with open("test_on_" + text_name + ".txt", "a") as f:
    f.write('checkpoint:' + checkpoint + '\n')
    f.write('test_num_sample:' + str(num_val) + '\n')
    f.write(result)