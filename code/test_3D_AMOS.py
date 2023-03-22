import argparse
import os
import shutil
from glob import glob

import torch

import pdb
import numpy as np
from medpy import metric
from networks.net_factory_3d import net_factory_3d
from monai.inferers import sliding_window_inference
from tqdm import tqdm
from monai.transforms import (
    AsDiscrete,
    Compose,
    SpatialPadd,
    ToMetaTensord,
    ToTensord,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandSpatialCropd,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    EnsureTyped,
)
from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
    set_track_meta,
)
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../datasets/AMOS', help='Name of Experiment')
parser.add_argument('--save_path', type=str,
                    default='../experiments/../model_best.pth', help='Models path')
parser.add_argument('--model', type=str,
                    default='unet_3D', help='model_name')
parser.add_argument('--num_classes', type=int,  default=16,
                    help='output channel of network')
parser.add_argument('--val_num', type=int, default=30,
                    help='labeled data')
parser.add_argument('--split_path', type=str,
                    default='../data/AMOS/dataset_semi_ct.json', help='Split info')
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=2, type=int, help="number of output channels")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
parser.add_argument("--use_ssl_pretrained", action="store_true", help="use self-supervised pretrained weights")
args = parser.parse_args()

val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        ScaleIntensityRanged(
            keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        SpatialPadd(keys=["image", "label"],
            spatial_size=(96, 96, 96)
        ),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        ToTensord(
            keys=["image", "label"],
        ),
    ]
)

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
    
def test_all_case_amos(net, val_loader, test_save_path=None, num_classes=4):
    total_metric = np.zeros((num_classes-1, 2))
    print("Validation begin")
    metric_cal = [[] for i in range(num_classes-1)]
    net.eval()
    with torch.no_grad():
        for step, batch in tqdm(enumerate(val_loader)):
            image, label = batch["image"].cuda(), batch["label"].squeeze(0)
            # spacing = (1,1,1)
            # ids = batch['image_meta_dict']['filename_or_obj'][0][-16:-7]
            with torch.cuda.amp.autocast():
                prediction = sliding_window_inference(image, (96, 96, 96), 4, net).cpu().numpy()
                prediction = torch.argmax(torch.tensor(prediction).cuda(), dim=1).cpu().numpy()
            for i in range(1, num_classes):
                temp_metric = cal_metric(label == i, prediction == i)
                total_metric[i-1, :] += temp_metric
                metric_cal[i-1].append(temp_metric)
            # print(np.mean(metric_cal, axis=0))
        
            # prediction = prediction.squeeze(0)
            # label = label.cpu().numpy().squeeze(0)
            # image = image.cpu().numpy().squeeze(0).squeeze(0)
            
            # pred_itk = sitk.GetImageFromArray(prediction.astype(np.uint8))
            # pred_itk.SetSpacing(spacing)
            # sitk.WriteImage(pred_itk, test_save_path +
            #                 "/{}_pred.nii.gz".format(ids))

            # img_itk = sitk.GetImageFromArray(image)
            # img_itk.SetSpacing(spacing)
            # sitk.WriteImage(img_itk, test_save_path +
            #                 "/{}_img.nii.gz".format(ids))

            # lab_itk = sitk.GetImageFromArray(label.astype(np.uint8))
            # lab_itk.SetSpacing(spacing)
            # sitk.WriteImage(lab_itk, test_save_path +
            #                 "/{}_lab.nii.gz".format(ids))
            
    return metric_cal

def Inference(FLAGS, test_save_path=None):
    class_list = ['SPL', 'RKI', 'LKI', 'GBL' ,'ESO', 'LIV', 'STO', 'AOR', 'IVC', 'PAN', 'RAG', 'LAG', 'DUO', 'BLA', 'PRO/UTE']
    num_classes = 16
    val_num = args.val_num
    net = net_factory_3d(net_type=args.model, in_chns=1, class_num=num_classes)

    snapshot_path = FLAGS.save_path
    net.load_state_dict(torch.load(snapshot_path))
    print("init weight from {}".format(snapshot_path))
    net.eval()
    
    datasets = FLAGS.root_path+'/dataset_semi_ct.json'
    val_files = load_decathlon_datalist(datasets, True, "validation")
    val_files = val_files[val_num:]
    
    val_ds = CacheDataset(data=val_files, transform=val_transforms, num_workers=6)
    valloader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)
    
    metric_cal = test_all_case_amos(net, valloader, test_save_path='../experiments/AMOS22/vis_results', num_classes=num_classes)
    # mean and std for all and each class
    mean_cal, std_cal = 0.0, 0.0
    class_mean, class_std = [], []
    for class_i in range(num_classes-1):
        _mean = np.mean(metric_cal[class_i], axis=0)
        _std = np.std(metric_cal[class_i], axis=0)
        mean_cal += _mean
        std_cal += _std
        class_mean.append(_mean)
        class_std.append(_std)
    mean_dsc, std_dsc = mean_cal[0]/(num_classes-1), std_cal[0]/(num_classes-1)
    mean_hd95, std_hd95 = mean_cal[1]/(num_classes-1), std_cal[1]/(num_classes-1)
    
    print('mean_dice : %f mean_hd95 : %f\n std_dice : %f std_hd95 : %f' % (mean_dsc, mean_hd95, std_dsc, std_hd95))
    for class_i in range(num_classes-1):
        print('{}_dice_mean: {}'.format(class_list[class_i], np.round(class_mean[class_i][0], 4)) + '    {}_hd95_mean: {}'.format(class_list[class_i], np.round(class_mean[class_i][1], 4)))
        print('{}_dice_std: {}'.format(class_list[class_i], np.round(class_std[class_i][0], 4)) + '    {}_hd95_std: {}'.format(class_list[class_i], np.round(class_std[class_i][1], 4)))
    return "Testing end"


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)
    print(metric)
