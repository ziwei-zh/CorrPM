import torch.nn as nn
import torch
from torch.nn import functional as F
import cv2
import numpy as np

class CriterionPoseEdge(nn.Module):
    def __init__(self, ignore_index=255, nclass=20, weight=None):
        super(CriterionPoseEdge, self).__init__()
        self.ignore_index = ignore_index
        self.nclass = nclass
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        self.pose_criterion = torch.nn.MSELoss()

    def parsing_loss(self, preds, target):
        '''
        target[0]: parsing
        target[1]:edge
        target[2]:pose
        '''
        self.h, self.w = target[0].size(1), target[0].size(2)
        h, w = target[0].size(1), target[0].size(2)

        pos_num = torch.sum(target[1] == 1, dtype=torch.float)
        neg_num = torch.sum(target[1] == 0, dtype=torch.float)
        weight_pos = neg_num / (pos_num + neg_num)
        weight_neg = pos_num / (pos_num + neg_num)
        weights = torch.tensor([weight_neg, weight_pos])
        loss = 0
        loss_parsing = 0
        loss_edge = 0
        loss_pose = 0

        # loss for parsing
        preds_parsing = preds[0]
        if isinstance(preds_parsing, list):
            for inst in preds_parsing:
                scale_pred = F.interpolate(input=inst, size=(h, w),
                                           mode='bilinear', align_corners=True)
                loss_ = self.criterion(scale_pred, target[0])
                loss_parsing += loss_
        else:
            scale_pred = F.interpolate(input=preds_parsing, size=(h, w),
                                       mode='bilinear', align_corners=True)
            loss_parsing += self.criterion(scale_pred, target[0])

        edge = target[1]
        pose = target[2]
        # loss for edge
        preds_edge = preds[1]
        if isinstance(preds_edge, list):
            for pred_edge in preds_edge:
                scale_pred = F.interpolate(input=pred_edge, size=(h, w),
                                           mode='bilinear', align_corners=True)
                loss_edge += F.cross_entropy(scale_pred, edge,
                                        weights.cuda(), ignore_index=self.ignore_index)
        else:
            scale_pred = F.interpolate(input=preds_edge, size=(h, w),
                                       mode='bilinear', align_corners=True)
            loss_edge += F.cross_entropy(scale_pred, edge,
                                    weights.cuda(), ignore_index=self.ignore_index)

        # loss for pose
        preds_pose = preds[2]
        pose_h, pose_w = pose.size(2), pose.size(3)
        if isinstance(preds_pose, list):
            for pred_pose in preds_pose:
                pred_pose = F.interpolate(input=pred_pose, size=(pose_h, pose_w), mode='bilinear', align_corners=True)
                loss_pose += self.pose_criterion(pred_pose, pose)
        else:
            scale_pred = F.interpolate(input=preds_pose, size=(h, w), mode='bilinear', align_corners=True)
            loss_pose += self.pose_criterion(scale_pred, pose)

        loss = loss_parsing + 2*loss_edge + 70*loss_pose
        return loss


    def forward(self, preds, target):
        loss = self.parsing_loss(preds, target)
        return loss

