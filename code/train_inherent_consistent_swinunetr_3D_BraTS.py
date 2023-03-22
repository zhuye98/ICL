import argparse
import logging
import os
import random
import shutil

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import pdb

from collections import OrderedDict
from dataloaders.brats2019 import (BraTS2019, RandomCrop,
                                   RandomRotFlip, ToTensor,
                                   TwoStreamBatchSampler)
from networks.net_factory_3d import net_factory_3d
from utils import losses
from val_3D import test_all_case_base

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/BraTS19', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='BraTS19/Inherent_Consistent_Learning', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='swinunetr_icl', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[96, 96, 96],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed') 
parser.add_argument('--labeled_num', type=int, default=25,
                    help='labeled data')
parser.add_argument('--labeled_bs', type=int, default=2,
                    help='labeled_batch_size per gpu')
parser.add_argument('--num_tries', type=str,  default='1',
                    help='number of experiments tryings')
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')

parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=2, type=int, help="number of output channels")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--dropout_path_rate", default=0.2, type=float, help="drop path rate")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
parser.add_argument("--use_ssl_pretrained", action="store_true", help="use self-supervised pretrained weights")

args = parser.parse_args()


def train(args, snapshot_path):

    base_lr = args.base_lr
    train_data_path = args.root_path
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    num_classes = args.num_classes
    
    model = net_factory_3d(net_type=args.model)
    
    if args.use_ssl_pretrained:
        try:

            model_dict = torch.load("../pretrained_models/model_swinvit.pt")
            state_dict = model_dict["state_dict"]

            # fix potential differences in state dict keys from pre-training to
            # fine-tuning
            if "module." in list(state_dict.keys())[0]:
                print("Tag 'module.' found in state dict - fixing!")
                for key in list(state_dict.keys()):
                    state_dict[key.replace("module.", "swinViT.")] = state_dict.pop(key)
            if "swin_vit" in list(state_dict.keys())[0]:
                print("Tag 'swin_vit' found in state dict - fixing!")
                for key in list(state_dict.keys()):
                    state_dict[key.replace("swin_vit", "swinViT")] = state_dict.pop(key)
            # We now load model weights, setting param `strict` to False, i.e.:
            # this load the encoder weights (Swin-ViT, SSL pre-trained), but leaves
            # the decoder weights untouched (CNN UNet decoder).
            model.load_state_dict(state_dict, strict=False)
            print("Using pretrained self-supervised Swin UNETR backbone weights !")
        except ValueError:
            raise ValueError("Self-supervised pre-trained weights not available for" + str(args.model_name))

    model.train()

    db_train = BraTS2019(base_dir=train_data_path,
                         split='train',
                         num=None,
                         transform=transforms.Compose([
                             RandomRotFlip(),
                             RandomCrop(args.patch_size),
                             ToTensor(),
                         ]))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    labeled_idxs = list(range(0, args.labeled_num))
    unlabeled_idxs = list(range(args.labeled_num, 250))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    aux_loss = losses.AuxLoss3D(num_classes)
    pse_loss = losses.PseudoSoftLoss3D(num_classes)

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
            
            outputs_soft = torch.softmax(outputs[0], dim=1)
            
            loss_ce = ce_loss(outputs[0], label_batch[:args.labeled_bs])
            loss_dice = dice_loss(outputs_soft, label_batch[:args.labeled_bs].unsqueeze(1))
            loss_aux = aux_loss(outputs[2], label_batch[:args.labeled_bs])
            loss_pse = pse_loss(outputs[3], outputs[1])
            loss_aux_consis = losses.softmax_mse_loss(outputs[3], outputs[4])
            loss = loss_dice + loss_ce + loss_aux + loss_pse + 10*loss_aux_consis
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('Info/lr', lr_, iter_num)
            writer.add_scalar('Loss/loss', loss, iter_num)
            writer.add_scalar('Loss/loss_ce', loss_ce, iter_num)
            writer.add_scalar('Loss/loss_dice', loss_dice, iter_num)
            writer.add_scalar('Loss/loss_aux', loss_aux, iter_num)
            writer.add_scalar('Loss/loss_pse', loss_pse, iter_num)
            writer.add_scalar('Loss/loss_aux_consis', 10*loss_aux_consis, iter_num)


            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, loss_aux: %f, loss_pse: %f, loss_aux_consis: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item(), loss_aux.item(), loss_pse.item(), 10*loss_aux_consis.item()))

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_cal = test_all_case_base(model, args.model, args.root_path, test_list="val_test.txt", num_classes=2, patch_size=args.patch_size, stride_xy=64, stride_z=64)
                
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

                logging.info('iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, mean_dsc, mean_hd95))
                logging.info('iteration %d : std_dice : %f std_hd95 : %f' % (iter_num, std_dsc, std_hd95))

                model.train()

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
    #logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
