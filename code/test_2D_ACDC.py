import argparse
from enum import Flag
import os
import shutil
from tracemalloc import Snapshot
from cv2 import UMAT_AUTO_STEP

import h5py
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from PIL import Image

from networks.net_factory import net_factory
from networks.vision_transformer import SwinUnet as ICL_seg
from networks.config import get_config


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Data path')
parser.add_argument('--save_path', type=str,
                    default='../experiments/../model_best.pth', help='Models path')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=3,
                    help='labeled data')


parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument(
    '--cfg', type=str, default="../code/configs/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                    'full: cache all data, '
                    'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')
args = parser.parse_args()
config = get_config(args)
            

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
        return 1, 

def test_single_volume(case, net, test_save_path, FLAGS, patch_size):
    h5f = h5py.File(FLAGS.root_path + "/volumes/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out_main = net(input)
            out = torch.argmax(torch.softmax(
                out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[0]), order=0)
            prediction[ind] = pred

    first_metric = calculate_metric_percase(prediction == 1, label == 1)
    second_metric = calculate_metric_percase(prediction == 2, label == 2)
    third_metric = calculate_metric_percase(prediction == 3, label == 3)

    # img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    # img_itk.SetSpacing((1, 1, 10))
    # prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    # prd_itk.SetSpacing((1, 1, 10))
    # lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    # lab_itk.SetSpacing((1, 1, 10))
    # sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    # sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    # sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric, second_metric, third_metric

def generate_prediction(case, net, test_save_path, FLAGS, case_name, patch_size=[256, 256]):
    h5f = h5py.File(FLAGS.root_path + "/volumes/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    
    prediction = np.zeros_like(label)
    img_save_path = test_save_path+'/'+case_name+'/images/'
    gt_save_path = test_save_path+'/'+case_name+'/groundt/'
    pred_save_path = test_save_path+'/'+case_name+'/predicts/'
    os.makedirs(img_save_path)
    os.makedirs(pred_save_path)
    os.makedirs(gt_save_path)
    for ind in range(image.shape[0]):
        slice_ori = image[ind, :, :]
        label_ori = label[ind, :, :]
        x, y = slice_ori.shape[0], slice_ori.shape[1]
        slice = zoom(slice_ori, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():

            #out_main = net(input, None)
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)

            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
            
            slice_ori = slice_ori * 255
            slice_ori = slice_ori.astype(np.uint8)
            pred = pred.astype(np.uint8)
            label_ori = label_ori.astype(np.uint8)
            
            im_slice = Image.fromarray(slice_ori)
            im_pred = Image.fromarray(pred)
            im_lab = Image.fromarray(label_ori)
            
            im_slice.save(img_save_path+str(ind)+'.png')
            #im_pred.save(pred_save_path+str(ind)+'.png')
            im_lab.save(gt_save_path+str(ind)+'.png')
            
            # img_itk = sitk.GetImageFromArray(image.astype(np.float32))
            # img_itk.SetSpacing((1, 1, 10))
            # prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
            # prd_itk.SetSpacing((1, 1, 10))
            # lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
            # lab_itk.SetSpacing((1, 1, 10))
            # sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
            # sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
            # sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
        # save pseudo label for each case
    first_metric = calculate_metric_percase(prediction == 1, label == 1)
    second_metric = calculate_metric_percase(prediction == 2, label == 2)
    third_metric = calculate_metric_percase(prediction == 3, label == 3)
    print('DSC: ',(first_metric[0]+second_metric[0]+third_metric[0])/3, 'HD95: ', (first_metric[1]+second_metric[1]+third_metric[1])/3)
        #file_name = root_path + '/slices_iter1/' + case_name + '_slice_' + str(ind+1) + '.h5'

def Inference(FLAGS, test_save_path='../experiments/ACDC/vis_results'):
    
    with open(FLAGS.root_path + '/val_test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])

    # model
    net = net_factory(net_type=FLAGS.model, in_chns=1,class_num=FLAGS.num_classes)
    # if FLAGS.model == 'Ours-ViT-Seg':
    #     net = ICL_seg(config, img_size=[args.patch_size[0], args.patch_size[1]], num_classes=args.num_classes).cuda()
    # else:
    #     net = net_factory(net_type=FLAGS.model, in_chns=1,class_num=FLAGS.num_classes)

    snapshot_path = FLAGS.save_path
    save_mode_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    net.load_state_dict(torch.load(snapshot_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()
    FLAGS.num_classes = 4

    metric_cal = [[] for i in range(FLAGS.num_classes-1)]
    for case in image_list:
        if FLAGS.model == 'swinunet':
            metric_i = test_single_volume(case, net, test_save_path, FLAGS, [224, 224])
        elif  FLAGS.model == 'unet':
            metric_i = test_single_volume(case, net, test_save_path, FLAGS, [256, 256])
        # if FLAGS.model in ('unet_urpc', 'unet_cct'):
        #     metric_i = test_single_volume_ds(case, net, FLAGS.num_classes)
        # elif FLAGS.model in ('unet', 'unet_aux', 'ViT_Seg','icl_unet'):
        #     metric_i = test_single_volume(case, net, test_save_path, FLAGS, FLAGS.patch_size)
        # else:
        #     metric_i = test_single_volume(case, net, test_save_path, FLAGS, FLAGS.patch_size)
        # visualized 
        #generate_prediction(case, net, test_save_path, FLAGS, case, FLAGS.patch_size)
        for class_i in range(FLAGS.num_classes-1):
            metric_cal[class_i].append(metric_i[class_i])
        
    mean_cal, std_cal = 0.0, 0.0
    class_mean, class_std = [], []
    for class_i in range(FLAGS.num_classes-1):
        _mean = np.mean(metric_cal[class_i], axis=0)
        _std = np.std(metric_cal[class_i], axis=0)
        mean_cal += _mean
        std_cal += _std
        class_mean.append(_mean)
        class_std.append(_std)

    mean_dsc, std_dsc = mean_cal[0]/(FLAGS.num_classes-1), std_cal[0]/(FLAGS.num_classes-1)
    mean_hd95, std_hd95 = mean_cal[1]/(FLAGS.num_classes-1), std_cal[1]/(FLAGS.num_classes-1)

    print("mean_dsc:", mean_dsc)
    print("std_dsc:", std_dsc)
    print("mean_hd95:", mean_hd95)
    print("std_hd95:", std_hd95)
    print("class_mean:", class_mean)
    print("class_std:", class_std)

    return 'Done!'


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)
    print(metric)
