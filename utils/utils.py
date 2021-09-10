# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Code written by Pavlo Molchanov and Hongxu Yin
# --------------------------------------------------------

import torch
import os
from torch import distributed, nn
import random
import numpy as np


def load_model_pytorch(model, load_model, gpu_n=0):
    print("=> loading checkpoint '{}'".format(load_model))

    checkpoint = torch.load(load_model, map_location=lambda storage, loc: storage.cuda(gpu_n))

    if 'state_dict' in checkpoint.keys():
        load_from = checkpoint['state_dict']
    else:
        load_from = checkpoint

    if 1:
        if 'module.' in list(model.state_dict().keys())[0]:
            if 'module.' not in list(load_from.keys())[0]:
                from collections import OrderedDict

                load_from = OrderedDict([("module.{}".format(k), v) for k, v in load_from.items()])

        if 'module.' not in list(model.state_dict().keys())[0]:
            if 'module.' in list(load_from.keys())[0]:
                from collections import OrderedDict

                load_from = OrderedDict([(k.replace("module.", ""), v) for k, v in load_from.items()])

    if 1:
        from collections import OrderedDict
        if list(load_from.items())[0][0][:2] == "1." and list(model.state_dict().items())[0][0][:2] != "1.":
            load_from = OrderedDict([(k[2:], v) for k, v in load_from.items()])

        load_from = OrderedDict([(k, v) for k, v in load_from.items() if "gate" not in k])

    # remove the prefix from the keys
    load_from2 = {}
    for k, v in load_from.items():
        if k == 'resnet34_8s.fc.weight':
            v = v[:, :, 0, 0]
        load_from2[k.replace("resnet34_8s.", "")] = v

    model.load_state_dict(load_from2, strict=True)

    epoch_from = -1
    if 'epoch' in checkpoint.keys():
        epoch_from = checkpoint['epoch']
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(load_model, epoch_from))


def create_folder(directory):
    # from https://stackoverflow.com/a/273227
    if not os.path.exists(directory):
        os.makedirs(directory)


random.seed(0)


def distributed_is_initialized():
    if distributed.is_available():
        if distributed.is_initialized():
            return True
    return False


def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return _alr


def lr_cosine_policy(base_lr, warmup_length, epochs):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn)


def beta_policy(mom_fn):
    def _alr(optimizer, iteration, epoch, param, indx):
        mom = mom_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group[param][indx] = mom

    return _alr


def mom_cosine_policy(base_beta, warmup_length, epochs):
    def _beta_fn(iteration, epoch):
        if epoch < warmup_length:
            beta = base_beta * (epoch + 1) / warmup_length
        else:
            beta = base_beta
        return beta

    return beta_policy(_beta_fn)


def clip(image_tensor, use_fp16=False):
    '''
    adjust the input based on mean and variance
    '''
    if use_fp16:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)
    return image_tensor


def denormalize(image_tensor, use_fp16=False):
    '''
    convert floats back to input
    '''
    if use_fp16:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c] * s + m, 0, 1)

    return image_tensor


class CustomPooling(nn.Module):
    def __init__(self, beta=1, r_0=0, dim=(-1, -2), mode=None):
        super(CustomPooling, self).__init__()
        self.r_0 = r_0

        self.dim = dim
        self.reset_parameters()
        self.mode = mode
        # self.device = device

        if self.mode is not None:
            # make a beta for each class --> size of tensor.
            self.beta = nn.Parameter(torch.nn.init.uniform_(torch.empty(3)), requires_grad=True)
            # self.cuda(self.device)
        else:
            self.beta = nn.Parameter(torch.nn.init.uniform_(torch.empty(1)), requires_grad=True)

    def reset_parameters(self, beta=None, r_0=None, dim=(-1, -2)):
        if beta is not None:
            init.zeros_(self.beta)
        if r_0 is not None:
            self.r_0 = r_0
        self.dim = dim

    def forward(self, x):
        '''
        :param x (tensor): tensor of shape [bs x K x h x w]
        :return logsumexp_torch (tensor): tensor of shape [bs x K], holding class scores per class
        '''

        if self.mode is None:
            const = self.r_0 + torch.exp(self.beta)
            _, _, h, w = x.shape
            average_constant = np.log(1. / (w * h))
            const = const.to('cuda')
            mod_out = const * x
            # logsumexp_torch = 1 / const * average_constant + 1 / const * torch.logsumexp(mod_out, dim=(-1, -2))
            logsumexp_torch = (average_constant + torch.logsumexp(mod_out, dim=(-1, -2)).to('cuda')) / const
            return logsumexp_torch
        else:
            const = self.r_0 + torch.exp(self.beta)
            _, d, h, w = x.shape

            average_constant = np.log(1. / (w * h))
            # mod_out = torch.zeros(x.shape)
            self.cuda(self.device)
            mod_out0 = const[0] * x[:, 0, :, :]
            mod_out1 = const[1] * x[:, 1, :, :]
            mod_out2 = const[2] * x[:, 2, :, :]

            mod_out = torch.cat((mod_out0.unsqueeze(1), mod_out1.unsqueeze(1), mod_out2.unsqueeze(1)), dim=1)
            # logsumexp_torch = 1 / const * average_constant + 1 / const * torch.logsumexp(mod_out, dim=(-1, -2))
            logsumexp_torch = (average_constant + torch.logsumexp(mod_out, dim=(-1, -2))) / const
            return logsumexp_torch