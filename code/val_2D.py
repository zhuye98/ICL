import h5py
import numpy as np
import torch
import pdb
from medpy import metric
from scipy.ndimage import zoom


visualized_list = ['patient019_frame01','patient013_frame02','patient066_frame01','patient066_frame02','patient087_frame01','patient087_frame02']

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 0, 373.128664
    elif pred.sum() == 0 and gt.sum() > 0:
        return 0, 373.128664
    elif pred.sum() == 0 and gt.sum() == 0:
        return 1, 0

def calculate_metric_dice_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        jc = metric.binary.jc(pred, gt)
        return dice, jc
    else:
        return 0, 0

def test_single_volume_ours(image, label, net, writer, iter_num, classes, patch_size=[224, 224]):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input, inference=True), dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()

            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    # visualize
    return metric_list

def evaluate_seg(pred, gt):
    #pdb.set_trace()
    pred_binary = (pred >= 0.5).float().cuda()
    pred_binary_inverse = (pred_binary == 0).float().cuda()

    gt_binary = (gt >= 0.5).float().cuda()
    gt_binary_inverse = (gt_binary == 0).float().cuda()

    MAE = torch.abs(pred_binary - gt_binary).mean().cuda(0)
    TP = pred_binary.mul(gt_binary).sum().cuda(0)
    FP = pred_binary.mul(gt_binary_inverse).sum().cuda(0)
    #TN = pred_binary_inverse.mul(gt_binary_inverse).sum().cuda(0)
    FN = pred_binary_inverse.mul(gt_binary).sum().cuda(0)

    if TP.item() == 0:
        TP = torch.Tensor([1]).cuda(0)
    # recall
    Recall = TP / (TP + FN)
    # Precision or positive predictive value
    Precision = TP / (TP + FP)
    # F1 score = Dice
    Dice = 2 * Precision * Recall / (Precision + Recall)
    # Overall accuracy
    #Accuracy = (TP + TN) / (TP + FP + FN + TN)
    # IoU for poly
    IoU_polyp = TP / (TP + FP + FN)

    return MAE.data.cpu().numpy().squeeze(), \
           Dice.data.cpu().numpy().squeeze(), \
           IoU_polyp.data.cpu().numpy().squeeze()


def generate_pseudo_labels(image, label, case_name, root_path, net, patch_size=[224, 224]):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    
    for ind in range(image.shape[0]):
        slice_ori = image[ind, :, :]
        x, y = slice_ori.shape[0], slice_ori.shape[1]
        slice = zoom(slice_ori, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input,None), dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()

            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
        # save pseudo label for each case
        file_name = root_path + '/slices_iter1/' + case_name + '_slice_' + str(ind+1) + '.h5'
        with h5py.File(file_name,'w') as f:
            f.create_dataset('image',data=slice_ori, compression="gzip")
            f.create_dataset('label',data=pred, compression="gzip")
    
    # visualize

def test_single_volume(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list

def test_single_volume_ds(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list
