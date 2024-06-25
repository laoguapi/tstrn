#!/usr/bin/env python
# encoding: utf-8
import torch
from Weight import Weight, Weight_1

# 高斯核生成函数
def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)

# def mmd_rbf_noaccelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
#     batch_size_sour = int(source.size()[0])
#     batch_size_tar = int(target.size()[0])
#     kernels = guassian_kernel(source, target,
#                               kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
#     XX = kernels[:batch_size_sour, :batch_size_sour]
#     YY = kernels[batch_size_tar:, batch_size_tar:]
#     # XY = kernels[:batch_size, batch_size:]
#     # YX = kernels[batch_size:, :batch_size]
#     loss = torch.trace(XX)/batch_size_sour**2 + torch.trace(YY)/batch_size_tar**2
#     #loss = torch.mean(XX + YY - XY -YX)
#     return loss

# 这个mmd距离的写法是网上的，前面注释掉的是代码本身的mmd函数，但是跑起来好像有问题，一直是一个数字不变
# https://blog.csdn.net/sinat_34173979/article/details/105876584
def mmd_rbf_noaccelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size_sour = int(source.size()[0])
    batch_size_tar = int(target.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size_sour, :batch_size_sour]
    YY = kernels[batch_size_tar:, batch_size_tar:]
    XY = kernels[:batch_size_sour, batch_size_tar:]
    YX = kernels[batch_size_tar:, :batch_size_sour]
    loss = torch.mean(XX + YY - XY -YX)
    return loss

# DSAN本身的lmmd函数
def lmmd(source, target, s_label, t_label, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = source.size()[0]
    weight_ss, weight_tt, weight_st = Weight.cal_weight(s_label, t_label, type='visual', class_num=5)
    weight_ss = torch.from_numpy(weight_ss).cuda()
    weight_tt = torch.from_numpy(weight_tt).cuda()
    weight_st = torch.from_numpy(weight_st).cuda()

    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = torch.Tensor([0]).cuda()
    if torch.sum(torch.isnan(sum(kernels))):
        return loss
    SS = kernels[:batch_size, :batch_size]
    TT = kernels[batch_size:, batch_size:]
    ST = kernels[:batch_size, batch_size:]

    loss += torch.sum( weight_ss * SS + weight_tt * TT - 2 * weight_st * ST )
    return loss

# 根据lmmd修改的nmmd，计算正负情感
# 想法是lmmd按照类别进行靠近，把类别固定成2（正负），就变成了nmmd
# 权重矩阵使用修改后的Weight_1（lmmd是Weight）
def nmmd(source, target, s_label, t_label, mode, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = source.size()[0]
    weight_ss, weight_tt, weight_st = Weight_1.cal_weight(s_label, t_label, mode, type='visual', class_num=2)
    weight_ss = torch.from_numpy(weight_ss).cuda()
    weight_tt = torch.from_numpy(weight_tt).cuda()
    weight_st = torch.from_numpy(weight_st).cuda()

    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = torch.Tensor([0]).cuda()
    if torch.sum(torch.isnan(sum(kernels))):
        return loss
    SS = kernels[:batch_size, :batch_size]
    TT = kernels[batch_size:, batch_size:]
    ST = kernels[:batch_size, batch_size:]

    loss += torch.sum( weight_ss * SS + weight_tt * TT - 2 * weight_st * ST )
    return loss

