from networks.unet_3D import unet_3D
from networks.unet_3D_icl import unet_3D_icl
from networks.vnet import VNet
from networks.VoxResNet import VoxResNet
from networks.attention_unet import Attention_UNet
from networks.nnunet import initialize_network
from networks.swinunetr import SwinUNETR
from networks.swinunetr_icl import SwinUNETR_icl
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/BraTS19', help='Name of Experiment')
parser.add_argument('--save_path', type=str,
                    default='../experiments/../model_best.pth', help='Models path')
parser.add_argument('--split_path', type=str,
                    default='../data/AMOS/dataset_semi_ct.json', help='Split info')
parser.add_argument('--model', type=str,
                    default='unet_3D', help='model_name')
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=2, type=int, help="number of output channels")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
parser.add_argument('--labeled_num', type=int, default=25,
                    help='labeled data')
parser.add_argument('--val_num', type=int, default=30,
                    help='labeled data')
parser.add_argument('--num_tries', type=str,  default='1',
                    help='number of experiments tryings')
args = parser.parse_args()

def net_factory_3d(net_type="unet_3D", in_chns=1, class_num=2):
    if net_type == "unet_3D":
        net = unet_3D(n_classes=class_num, in_channels=in_chns).cuda()
    elif net_type == 'unet_3D_icl':
        net = unet_3D_icl(n_classes=class_num, in_channels=in_chns).cuda()
    elif net_type == "swinunetr":
        net = SwinUNETR(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=args.in_channels,
            out_channels=args.num_classes,
            feature_size=args.feature_size,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=args.dropout_path_rate,
            use_checkpoint=args.use_checkpoint).cuda()
    elif net_type == "swinunetr_icl":
        net = SwinUNETR_icl(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=args.in_channels,
            out_channels=args.num_classes,
            feature_size=args.feature_size,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=args.dropout_path_rate,
            use_checkpoint=args.use_checkpoint).cuda()
    elif net_type == "nnUNet":
        net = initialize_network(num_classes=class_num).cuda()
    else:
        net = None
    return net
