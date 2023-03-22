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
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from collections import OrderedDict

from dataloaders.dataset import BaseDataSets, RandomGenerator, TwoStreamBatchSampler
from networks.net_factory import net_factory
from utils import losses
from val_2D import test_single_volume_ours

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/Inherent_Consistent_Learning', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='icl_unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--labeled_num', type=int, default=50,
                    help='labeled data')
parser.add_argument('--num_tries', type=str,  default='1',
                    help='number of experiments tryings')

parser.add_argument('--labeled_bs', type=int, default=8,
                    help='labeled_batch_size per gpu')
args = parser.parse_args()


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)

    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val_test")

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)
    
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    aux_loss = losses.AuxLoss(num_classes, resize=args.patch_size)
    pse_loss = losses.PseudoSoftLoss(num_classes, resize=args.patch_size)
    
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            outputs = model(volume_batch[:args.labeled_bs], volume_batch[args.labeled_bs:])

            loss_ce = ce_loss(outputs[0], label_batch[:args.labeled_bs].long())
            loss_dice = dice_loss(outputs[0], label_batch[:args.labeled_bs].unsqueeze(1), softmax=True)
            loss_aux = aux_loss(outputs[2], label_batch[:args.labeled_bs])

            loss_pse = pse_loss(outputs[3], outputs[1])
            loss_aux_consis = losses.softmax_mse_loss(outputs[3], outputs[4])

            loss_seg = loss_ce + loss_dice
            loss = loss_seg + loss_aux + loss_pse + 50*loss_aux_consis
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('Info/lr', lr_, iter_num)
            writer.add_scalar('Loss/loss', loss, iter_num)
            writer.add_scalar('Loss/loss_seg', loss_seg, iter_num)
            writer.add_scalar('Loss/loss_aux', loss_aux, iter_num)
            writer.add_scalar('Loss/loss_pse', loss_pse, iter_num)
            writer.add_scalar('Loss/loss_aux_consis', 50*loss_aux_consis, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, loss_pse: %f, loss_aux: %f, loss_aux_consis: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item(), loss_pse.item(), loss_aux.item(), 50*loss_aux_consis.item()))

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_cal = [[] for i in range(num_classes-1)]
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume_ours(
                        sampled_batch["image"], sampled_batch["label"], model, writer, iter_num, classes=num_classes, patch_size=args.patch_size)
                    # cal mean and variance
                    for class_i in range(num_classes-1):
                        metric_cal[class_i].append(metric_i[class_i])

                # mean and std for all and each classs
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

                    save_dict = OrderedDict()
                    for key, val in model.state_dict().items():
                        if not 'sspa' in key and not 'uscl' in key:
                            save_dict.update({key:val})
                    torch.save(save_dict, save_best)
                    logging.info('saving best model at iter {}'.format(iter_num))
                
                # saving and logging metric values 
                writer.add_scalar('metric_all/mean_dice', mean_dsc, iter_num)
                writer.add_scalar('metric_all/mean_hd95', mean_hd95, iter_num)
                writer.add_scalar('metric_all/std_dice', std_dsc, iter_num)
                writer.add_scalar('metric_all/std_hd95', std_hd95, iter_num)

                writer.add_scalar('metric_class_RV/mean_dice', class_mean[0][0], iter_num)
                writer.add_scalar('metric_class_RV/mean_hd95', class_mean[0][1], iter_num)
                writer.add_scalar('metric_class_RV/std_dice', class_std[0][0], iter_num)
                writer.add_scalar('metric_class_RV/std_hd95', class_std[0][1], iter_num)

                writer.add_scalar('metric_class_Myo/mean_dice', class_mean[1][0], iter_num)
                writer.add_scalar('metric_class_Myo/mean_hd95', class_mean[1][1], iter_num)
                writer.add_scalar('metric_class_Myo/std_dice', class_std[1][0], iter_num)
                writer.add_scalar('metric_class_Myo/std_hd95', class_std[1][1], iter_num)

                writer.add_scalar('metric_class_LV/mean_dice', class_mean[2][0], iter_num)
                writer.add_scalar('metric_class_LV/mean_hd95', class_mean[2][1], iter_num)
                writer.add_scalar('metric_class_LV/std_dice', class_std[2][0], iter_num)
                writer.add_scalar('metric_class_LV/std_hd95', class_std[2][1], iter_num)

                logging.info('iteration %d : mean_dice : %f  mean_hd95 : %f' % (iter_num, mean_dsc, mean_hd95))
                logging.info('iteration %d : std_dice : %f  std_hd95 : %f' % (iter_num, std_dsc, std_hd95))
                logging.info('iteration %d : RV_mean_dice : %f  RV_mean_hd95 : %f' % (iter_num, class_mean[0][0], class_mean[0][1]))
                logging.info('iteration %d : RV_std_dice : %f  RV_std_hd95 : %f' % (iter_num, class_std[0][0], class_std[0][1]))
                logging.info('iteration %d : Myo_mean_dice : %f  Myo_mean_hd95 : %f' % (iter_num, class_mean[1][0], class_mean[1][1]))
                logging.info('iteration %d : Myo_std_dice : %f  Myo_std_hd95 : %f' % (iter_num, class_std[1][0], class_std[1][1]))
                logging.info('iteration %d : LV_mean_dice : %f  LV_mean_hd95 : %f' % (iter_num, class_mean[2][0], class_mean[2][1]))
                logging.info('iteration %d : LV_std_dice : %f  LV_std_hd95 : %f' % (iter_num, class_std[2][0], class_std[2][1]))
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

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
