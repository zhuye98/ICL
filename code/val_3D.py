import math
from glob import glob

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from medpy import metric
from tqdm import tqdm
import pdb
from monai.inferers import sliding_window_inference

def test_single_case_base(net, net_type, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2, w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2, h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2, d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                               (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0],
                                   ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(
                    test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    if net_type in ('swinunetr_icl', 'unet_3D_icl'):
                        y1 = net(test_patch, inference=True)
                    else:
                        y1 = net(test_patch)
                    y = torch.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)

    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,
                              hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        score_map = score_map[:, wl_pad:wl_pad +
                              w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
    return label_map


def cal_metric(gt, pred):
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


def test_all_case_base(net, net_type, base_dir, test_list="full_test.list", num_classes=4, patch_size=(48, 160, 160), stride_xy=32, stride_z=24):
    with open(base_dir + '/{}'.format(test_list), 'r') as f:
        image_list = f.readlines()
    image_list = [base_dir + "/data/{}.h5".format(
        item.replace('\n', '').split(",")[0]) for item in image_list]
    total_metric = np.zeros((num_classes-1, 2))
    print("Validation begin")
    metric_cal = [[] for i in range(num_classes-1)]
    for image_path in tqdm(image_list):
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        prediction = test_single_case_base(
            net, net_type, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        for i in range(1, num_classes):
            temp_metric = cal_metric(label == i, prediction == i)
            total_metric[i-1, :] += temp_metric
            metric_cal[i-1].append(temp_metric)
    return metric_cal

def test_all_case_amos(net, net_type, val_loader, num_classes=4):
    total_metric = np.zeros((num_classes-1, 2))
    print("Validation begin")
    metric_cal = [[] for i in range(num_classes-1)]
    net.eval()
    with torch.no_grad():
        for step, batch in enumerate(tqdm(val_loader)):
            image, label = batch["image"].cuda(), batch["label"].squeeze(0)
            with torch.cuda.amp.autocast():
                if net_type == 'unet_3D_icl':
                    prediction = sliding_window_inference(image, (96, 96, 96), 4, net, inference=True).cpu().numpy()
                else:
                    prediction = sliding_window_inference(image, (96, 96, 96), 4, net).cpu().numpy()
                prediction = torch.argmax(torch.tensor(prediction).cuda(), dim=1).cpu().numpy()
            for i in range(1, num_classes):
                temp_metric = cal_metric(label == i, prediction == i)
                total_metric[i-1, :] += temp_metric
                metric_cal[i-1].append(temp_metric)
    return metric_cal