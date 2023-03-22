import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.modules.loss import CrossEntropyLoss


import pdb

def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def dice_loss1(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def entropy_loss(p, C=2):
    # p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1) / \
        torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent


def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1*torch.sum(p * torch.log(p + 1e-6), dim=1,
                       keepdim=True)/torch.tensor(np.log(C)).cuda()
    return ent


def softmax_mse_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    #assert input_logits.size() == target_logits.size()
    loss = 0.0
    if sigmoid:
        input_softmax = torch.sigmoid(input_logits)
        target_softmax = torch.sigmoid(target_logits)
    else:
        for i in range(len(input_logits)):
            target_logit = target_logits[i].clone().detach()
            target_softmax = F.softmax(target_logit, dim=1)
            
            input_softmax = F.softmax(input_logits[i], dim=1)
        
            mse_loss = torch.mean((input_softmax-target_softmax)**2)
            loss += mse_loss
    return loss/len(input_logits)


def softmax_kl_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_log_softmax = torch.log(torch.sigmoid(input_logits))
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_log_softmax = F.log_softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='mean')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div


def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    #assert input1.size() == input2.size()
    loss = 0.0
    for i in range(len(input1)):
        input = input1[0].clone().detach()
        loss += torch.mean((input - input2[i])**2)
    return loss/3

class EAMLoss(nn.Module):
    def __init__(self,n_classes):
        super(EAMLoss, self).__init__()
        self.n_classes = n_classes
        self.ce_loss=CrossEntropyLoss(ignore_index=255)

    def forward(self, feat_maps, labels):
        loss_ce = 0.0
        #resize_ = [[14, 14], [28, 28], [56, 56]]
        resize = [224, 224]
        gt = labels.clone() # [bs, 224, 224]
        length = len(feat_maps)
        for idx in range(length):
            # feat_maps[idx]=[18, 14, 14, 14], pseudo_label=[18, 1, 14, 14] 
            feat_map_ = nn.functional.interpolate(feat_maps[idx].clone().float(), size=resize, mode='bilinear')
            loss_ce += self.ce_loss(feat_map_, gt.long())
        eam_loss = loss_ce/3.0
        return eam_loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

def cos_sim_loss(input1, input2):
    loss = 0.0
    for i in range(len(input1)):
        input_1 = input1[0].reshape(-1)
        input_2 = input2[0].reshape(-1)
        loss += (1-torch.cosine_similarity(input_1, input_2, dim=-1))

    return loss/3


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes
    
class AuxLoss(nn.Module):
    def __init__(self,n_classes,resize=[224, 224]):
        super(AuxLoss, self).__init__()
        self.n_classes = n_classes
        self.ce_loss=CrossEntropyLoss()
        self.dice_loss=DiceLoss(n_classes)
        self.resize = resize

    def forward(self, feat_maps, labels):
        loss_ce, loss_dice = 0.0, 0.0
        #resize_ = [[14, 14], [28, 28], [56, 56]]
        gt = labels.clone() # [bs, 224, 224]
        length = len(feat_maps)
        for idx in range(length):
            feat_map_ = nn.functional.interpolate(feat_maps[idx].clone().float(), size=self.resize, mode='bilinear')
            loss_ce += self.ce_loss(feat_map_, gt.long())
            loss_dice += self.dice_loss(feat_map_, gt.unsqueeze(1), softmax=True)
        aux_loss = loss_ce/length + loss_dice/length
        return aux_loss


class AuxLoss3D(nn.Module):
    def __init__(self,n_classes):
        super(AuxLoss3D, self).__init__()
        self.n_classes = n_classes
        self.ce_loss=CrossEntropyLoss()
        self.dice_loss=DiceLoss(n_classes)

    def forward(self, feat_maps, labels):
        loss_ce, loss_dice = 0.0, 0.0
        resize = [96, 96, 96]
        gt = labels.clone() # [bs, 224, 224]
        length = len(feat_maps)
        for idx in range(length):
            feat_map_ = nn.functional.interpolate(feat_maps[idx].clone().float(), size=resize, mode='trilinear')
            loss_ce += self.ce_loss(feat_map_, gt.long())
            loss_dice += self.dice_loss(feat_map_, gt.unsqueeze(1), softmax=True)
        aux_loss = loss_ce/length + loss_dice/length
        return aux_loss

class PseudoSoftLoss(nn.Module):
    def __init__(self,n_classes, resize=[224, 224]):
        super(PseudoSoftLoss, self).__init__()
        self.resize = resize
    def forward(self, feat_maps, predicts):
        loss_dice = 0.0
        length = len(feat_maps)
        de_predicts = predicts.clone().detach()
        for idx in range(length):
            feat_map_ = nn.functional.interpolate(feat_maps[idx].clone().float(), size=self.resize, mode='bilinear')
            loss_dice += softmax_dice_loss(feat_map_.clone().float(), de_predicts)
        pse_loss = loss_dice/length
        return pse_loss

class PseudoSoftLoss3D(nn.Module):
    def __init__(self,n_classes):
        super(PseudoSoftLoss3D, self).__init__()
    def forward(self, feat_maps, predicts):
        loss_dice = 0.0
        resize = [96, 96, 96]
        length = len(feat_maps)
        de_predicts = predicts.clone().detach()
        for idx in range(length):
            feat_map_ = nn.functional.interpolate(feat_maps[idx].clone().float(), size=resize, mode='trilinear')
            loss_dice += softmax_dice_loss(feat_map_.clone().float(), de_predicts)
        pse_loss = loss_dice/length
        return pse_loss


class KD_Loss(nn.Module):
    def __init__(self, subscale=0.0625):
        super(KD_Loss, self).__init__()
        self.subscale = int(1 / subscale)

    def forward(self, guidance_1, guidance_2):
        _, N, _ = guidance_1[0].shape
        kl_loss = 0.0
        length = len(guidance_1)
        for i in range(length):
            for j in range(N):
                prob_1 = torch.nn.functional.softmax(guidance_1[i][:, j, :] / 2.0, dim=1)
                prob_2 = torch.nn.functional.softmax(guidance_2[i][:, j, :] / 2.0, dim=1)
                loss = (torch.sum(prob_1 * torch.log(prob_1 / prob_2)) + torch.sum(prob_2 * torch.log(prob_2 / prob_1))) / 2.0
            kl_loss += loss
        kl_loss = kl_loss/float(length)
        return kl_loss


def entropy_minmization(p):
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1)
    ent = torch.mean(y1)

    return ent


def entropy_map(p):
    ent_map = -1*torch.sum(p * torch.log(p + 1e-6), dim=1,
                           keepdim=True)
    return ent_map


def compute_kl_loss(p, q):
    loss = 0.0
    for N in range(len(q)):
        p_loss = F.kl_div(F.log_softmax(p[N], dim=-1),
                        F.softmax(q[N], dim=-1), reduction='none')
        q_loss = F.kl_div(F.log_softmax(q[N], dim=-1),
                        F.softmax(p[N], dim=-1), reduction='none')

    # Using function "sum" and "mean" are depending on your task
        p_loss = p_loss.mean()
        q_loss = q_loss.mean()
        loss += (p_loss + q_loss) / 2
    return loss/len(q)
