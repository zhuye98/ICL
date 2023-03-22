import torch
import torch.nn as nn
import pdb


def visualized(lab_outputs, unlab_outputs, lab_volume_batch, unlab_volume_batch, label_batch, iter_num, lab_idx, unlab_idx, writer):
    # original labeled image
    image_ori_lab = lab_volume_batch[lab_idx, 0:1, :, :]
    writer.add_image('visualized_Img_Mask/Image_lab', image_ori_lab, iter_num)

    # original unlabled image
    image_ori_unlab = unlab_volume_batch[unlab_idx, 0:1, :, :]
    writer.add_image('visualized_Img_Mask/Image_unlab', image_ori_unlab, iter_num)

    # ground truth 
    labs = label_batch[lab_idx, ...].unsqueeze(0) * 50
    labs_scale1 = nn.functional.interpolate(labs.clone().float().unsqueeze(0), size=[14, 14])
    labs_scale2 = nn.functional.interpolate(labs.clone().float().unsqueeze(0), size=[28, 28])
    labs_scale3 = nn.functional.interpolate(labs.clone().float().unsqueeze(0), size=[56, 56])
    writer.add_image('visualized_Img_Mask/GroundTruth_lab', labs, iter_num)
    writer.add_image('visualized/GroundTruth_lab_scale1', labs_scale1[0], iter_num)
    writer.add_image('visualized/GroundTruth_lab_scale2', labs_scale2[0], iter_num)
    writer.add_image('visualized/GroundTruth_lab_scale3', labs_scale3[0], iter_num)

    # aux feature maps
    aux_scale1 = torch.argmax(torch.softmax(lab_outputs[2][0], dim=1), dim=1, keepdim=True)
    aux_scale2 = torch.argmax(torch.softmax(lab_outputs[2][1], dim=1), dim=1, keepdim=True)
    aux_scale3 = torch.argmax(torch.softmax(lab_outputs[2][2], dim=1), dim=1, keepdim=True)
    writer.add_image('visualized/Aux_feat_scale1', aux_scale1[lab_idx] * 50, iter_num)
    writer.add_image('visualized/Aux_feat_scale2', aux_scale2[lab_idx] * 50, iter_num)
    writer.add_image('visualized/Aux_feat_scale3', aux_scale3[lab_idx] * 50, iter_num)
    
    # pseudo feature maps
    pseudo_scale1 = torch.argmax(torch.softmax(unlab_outputs[3][0], dim=1), dim=1, keepdim=True)
    pseudo_scale2 = torch.argmax(torch.softmax(unlab_outputs[3][1], dim=1), dim=1, keepdim=True)
    pseudo_scale3 = torch.argmax(torch.softmax(unlab_outputs[3][2], dim=1), dim=1, keepdim=True)
    writer.add_image('visualized/Pseudo_feat_scale1', pseudo_scale1[unlab_idx-8] * 50, iter_num)
    writer.add_image('visualized/Pseudo_feat_scale2', pseudo_scale2[unlab_idx-8] * 50, iter_num)
    writer.add_image('visualized/Pseudo_feat_scale3', pseudo_scale3[unlab_idx-8] * 50, iter_num)
    # predictions_lab
    predicts_lab = torch.argmax(torch.softmax(lab_outputs[0], dim=1), dim=1, keepdim=True)
    predicts_lab_scale1 = nn.functional.interpolate(predicts_lab.clone().float(), size=[14, 14])
    predicts_lab_scale2 = nn.functional.interpolate(predicts_lab.clone().float(), size=[28, 28])
    predicts_lab_scale3 = nn.functional.interpolate(predicts_lab.clone().float(), size=[56, 56])
    writer.add_image('visualized_Img_Mask/Predicts_lab',predicts_lab[lab_idx, ...] * 50, iter_num)
    writer.add_image('visualized/Predicts_lab_scale1',predicts_lab_scale1[lab_idx] * 50, iter_num)
    writer.add_image('visualized/Predicts_lab_scale2',predicts_lab_scale2[lab_idx] * 50, iter_num)
    writer.add_image('visualized/Predicts_lab_scale3',predicts_lab_scale3[lab_idx] * 50, iter_num)
    # predictions_unlab
    predicts_unlab = torch.argmax(torch.softmax(unlab_outputs[1], dim=1), dim=1, keepdim=True)
    predicts_unlab_scale1 = nn.functional.interpolate(predicts_unlab.clone().float(), size=[14, 14])
    predicts_unlab_scale2 = nn.functional.interpolate(predicts_unlab.clone().float(), size=[28, 28])
    predicts_unlab_scale3 = nn.functional.interpolate(predicts_unlab.clone().float(), size=[56, 56])
    writer.add_image('visualized_Img_Mask/Predicts_unlab',predicts_unlab[unlab_idx-8, ...] * 50, iter_num)
    writer.add_image('visualized/Predicts_unlab_scale1',predicts_unlab_scale1[unlab_idx-8] * 50, iter_num)
    writer.add_image('visualized/Predicts_unlab_scale2',predicts_unlab_scale2[unlab_idx-8] * 50, iter_num)
    writer.add_image('visualized/Predicts_unlab_scale3',predicts_unlab_scale3[unlab_idx-8] * 50, iter_num)
    