import argparse
import math

import h5py
import numpy as np
import torch
from medpy import metric
from tqdm import tqdm

from networks.net_factory_3d import net_factory_3d

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/BraTS19', help='Name of Experiment')
parser.add_argument('--save_path', type=str,
                    default='../experiments/../model_best.pth', help='Models path')
parser.add_argument('--model', type=str,
                    default='unet_3D', help='model_name')
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')

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


def test_all_case(net, base_dir, method="unet_3D", test_list="full_test.list", num_classes=4, patch_size=(48, 160, 160), stride_xy=32, stride_z=24, test_save_path=None):
    with open(base_dir + '/{}'.format(test_list), 'r') as f:
        image_list = f.readlines()
    image_list = [base_dir + "/data/{}.h5".format(
        item.replace('\n', '').split(",")[0]) for item in image_list]
    total_metric = np.zeros((num_classes-1, 2))

    metric_cal = [[] for i in range(num_classes-1)]

    print("Testing begin")
   
    for image_path in tqdm(image_list):
        ids = image_path.split("/")[-1].replace(".h5", "")
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        
        prediction = test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        metric = calculate_metric_percase(label == 1, prediction == 1)
        metric_cal[0].append(metric)


        # f.writelines("{},{},{},{},{}\n".format(
        #     ids, metric[0], metric[1], metric[2], metric[3]))

        # pred_itk = sitk.GetImageFromArray(prediction.astype(np.uint8))
        # pred_itk.SetSpacing((1.0, 1.0, 1.0))
        # sitk.WriteImage(pred_itk, test_save_path +
        #                 "/{}_pred.nii.gz".format(ids))

        # img_itk = sitk.GetImageFromArray(image)
        # img_itk.SetSpacing((1.0, 1.0, 1.0))
        # sitk.WriteImage(img_itk, test_save_path +
        #                 "/{}_img.nii.gz".format(ids))

        # lab_itk = sitk.GetImageFromArray(label.astype(np.uint8))
        # lab_itk.SetSpacing((1.0, 1.0, 1.0))
        # sitk.WriteImage(lab_itk, test_save_path +
        #                 "/{}_lab.nii.gz".format(ids))
        # f.writelines("Mean metrics,{},{},{},{}".format(total_metric[0, 0] / len(image_list), total_metric[0, 1] / len(
        #     image_list), total_metric[0, 2] / len(image_list), total_metric[0, 3] / len(image_list)))

    return metric_cal

def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1):
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

def Inference(FLAGS, test_save_path='../experiments/BraTS19/vis_results'):
    
    num_classes = 2
    net = net_factory_3d(net_type=args.model, in_chns=1, class_num=num_classes)
    
    snapshot_path = FLAGS.save_path
    net.load_state_dict(torch.load(snapshot_path))
    print("init weight from {}".format(snapshot_path))
    
    
    avg_metric = test_all_case(net, base_dir=FLAGS.root_path, method=FLAGS.model, test_list="val_test.txt", num_classes=num_classes,
                               patch_size=(96, 96, 96), stride_xy=64, stride_z=64, test_save_path=test_save_path)
    mean_cal, std_cal = 0.0, 0.0
    class_mean, class_std = [], []
    for class_i in range(num_classes-1):
        _mean = np.mean(avg_metric[class_i], axis=0)
        _std = np.std(avg_metric[class_i], axis=0)
        mean_cal += _mean
        std_cal += _std
        class_mean.append(_mean)
        class_std.append(_std)

    mean_dsc, std_dsc = mean_cal[0]/(num_classes-1), std_cal[0]/(num_classes-1)
    mean_hd95, std_hd95 = mean_cal[1]/(num_classes-1), std_cal[1]/(num_classes-1)
    print("mean_dsc:", mean_dsc)
    print("std_dsc:", std_dsc)
    print("mean_hd95:", mean_hd95)
    print("std_hd95:", std_hd95)
    return "Testing end"


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return (0, 373.128664)
    elif pred.sum() == 0 and gt.sum() > 0:
        return (0, 373.128664)
    elif pred.sum() == 0 and gt.sum() == 0:
        return (1, 0)

if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)
    print(metric)
