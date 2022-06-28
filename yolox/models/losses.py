#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        iou = (area_i) / (area_p + area_g - area_i + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_i) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)
        elif self.loss_type == "eiou":
            c_tl = torch.min((pred[:,:2]-pred[:,2:]/2),(target[:,:2]-target[:,2:]/2))
            c_br = torch.max((pred[:,:2]+pred[:,2:]/2),(target[:,:2]+target[:,2:]/2))
            convex_dis = torch.pow(c_br[:,0]-c_tl[:,0],2) + torch.pow(c_br[:,1]-c_tl[:,1],2)
            center_dis = (torch.pow(pred[:,0]-target[:,0],2) + torch.pow(pred[:,1]-target[:,1],2))

            dis_w = torch.pow(pred[:,2] - target[:,2],2)
            dis_h = torch.pow(pred[:,3] - target[:,3],2)
            c_w = torch.pow(c_tl[:,0]-c_br[:,0],2)
            c_h = torch.pow(c_tl[:,1]-c_br[:,1],2)

            eiou = iou - (center_dis/convex_dis.clamp(1e-16)) - (dis_w/c_w.clamp(1e-16)) - (dis_h/c_h.clamp(1e-16))
            loss = 1-eiou.clamp(min=-1.0,max=1.0)
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss
