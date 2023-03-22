import argparse
import logging
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from tqdm import tqdm
import pdb

from networks.net_factory_3d import net_factory_3d
from utils import losses
from val_3D import test_all_case_amos
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
                    default='../data/AMOS', help='Name of Experiment')
parser.add_argument('--split_path', type=str,
                    default='../data/AMOS/dataset_semi_ct.json', help='Split info')
parser.add_argument('--exp', type=str,
                    default='AMOS22/UNet3D-FullyS', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet_3D', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=60000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.02,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[96, 96, 96],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--labeled_num', type=int, default=15,
                    help='labeled data')
parser.add_argument('--val_num', type=int, default=30,
                    help='labeled data')
parser.add_argument('--num_tries', type=str,  default='1',
                    help='number of experiments tryings')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        RandSpatialCropd(keys=["image", "label"],
            roi_size=(96, 96, 96),
            random_size=False,      
        ),
        SpatialPadd(keys=["image", "label"],
            spatial_size=(96, 96, 96)
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob=0.10,
        ),
        RandRotate90d(
            keys=["image", "label"],
            prob=0.10,
            max_k=3,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        ),
        ToTensord(
            keys=["image"],
        ),
    ]
)
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
class_list = ['SPL', 'RKI', 'LKI', 'GBL' ,'ESO', 'LIV', 'STO', 'AOR', 'IVC', 'PAN', 'RAG', 'LAG', 'DUO', 'BLA', 'PRO/UTE']
def train(args, snapshot_path):
    base_lr = args.base_lr
    max_iterations = args.max_iterations
    labeled_num = args.labeled_num
    val_num = args.val_num
    splits = args.split_path
    num_classes = 15+1
    model = net_factory_3d(net_type=args.model, in_chns=1, class_num=num_classes)
    datalist = load_decathlon_datalist(splits, True, "training")
    val_files = load_decathlon_datalist(splits, True, "validation")
    datalist = datalist[:labeled_num]
    val_files = val_files[:val_num]
    train_ds = CacheDataset(
        data=datalist,
        transform=train_transforms,
        # cache_num=24,
        # cache_rate=1.0,
        num_workers=6,
    )    
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    train_loader = ThreadDataLoader(train_ds, num_workers=6, batch_size=4, shuffle=True, worker_init_fn=worker_init_fn)
    
    val_ds = CacheDataset(data=val_files, transform=val_transforms, num_workers=6)
    val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)

    model.train()
    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(train_loader)))

    iter_num = 0
    max_epoch = max_iterations // len(train_loader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(train_loader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)

            loss_ce = ce_loss(outputs, label_batch.squeeze(1).long())
            loss_dice = dice_loss(outputs_soft, label_batch.long())
            loss = 0.5 * (loss_dice + loss_ce)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('Info/lr', lr_, iter_num)
            writer.add_scalar('Loss/loss', loss.item(), iter_num)
            writer.add_scalar('Loss/loss_ce', loss_ce.item(), iter_num)
            writer.add_scalar('Loss/loss_dice', loss_dice.item(), iter_num)
            logging.info('\niteration %d : loss : %f, loss_ce: %f, loss_dice: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

            # if iter_num % 20 == 0:
            #     image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
            #         3, 0, 1, 2).repeat(1, 3, 1, 1)
            #     grid_image = make_grid(image, 5, normalize=True)
            #     writer.add_image('train/Image', grid_image, iter_num)

            #     image = outputs_soft[0, 1:2, :, :, 20:61:10].permute(
            #         3, 0, 1, 2).repeat(1, 3, 1, 1)
            #     grid_image = make_grid(image, 5, normalize=False)
            #     writer.add_image('train/Predicted_label',
            #                      grid_image, iter_num)

            #     image = label_batch[0, :, :, 20:61:10].unsqueeze(
            #         0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
            #     grid_image = make_grid(image, 5, normalize=False)
            #     writer.add_image('train/Groundtruth_label',
            #                      grid_image, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_cal = test_all_case_amos(model, val_loader, num_classes=16)
                
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

                # saving the best model
                if mean_dsc > best_performance:
                    best_performance = mean_dsc
                    save_best = os.path.join(snapshot_path+'/model','model_best.pth')
                    torch.save(model.state_dict(), save_best)
                    logging.info('saving best model at iter {}'.format(iter_num))
                    
                # saving and logging metric values 
                for class_i in range(num_classes-1):
                    writer.add_scalar('metric/{}_dice'.format(class_list[class_i]),
                                      class_mean[class_i][0], iter_num)
                    writer.add_scalar('metric/{}_hd95'.format(class_list[class_i]),
                                      class_mean[class_i][1], iter_num)
                writer.add_scalar('metric_all/mean_dice', mean_dsc, iter_num)
                writer.add_scalar('metric_all/mean_hd95', mean_hd95, iter_num)
                writer.add_scalar('metric_all/std_dice', std_dsc, iter_num)
                writer.add_scalar('metric_all/std_hd95', std_hd95, iter_num)

                logging.info('iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, mean_dsc, mean_hd95))
                for class_i in range(num_classes-1):
                    logging.info('{}_dice: {}'.format(class_list[class_i], np.round(class_mean[class_i][0], 4)) + '    {}_hd95: {}'.format(class_list[class_i], np.round(class_mean[class_i][1], 4)))
                model.train()


                # if iter_num % 3000 == 0:
                #     save_mode_path = os.path.join(
                #         snapshot_path, 'iter_' + str(iter_num) + '.pth')
                #     torch.save(model.state_dict(), save_mode_path)
                #     logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../experiments/{}_{}_labeled/{}_exp_{}".format(
        args.exp, args.labeled_num, args.model, args.num_tries)
    if not os.path.exists(snapshot_path+'/model'):
        os.makedirs(snapshot_path+'/model')

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    #logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
