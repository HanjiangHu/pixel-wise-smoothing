# this file is based on code publicly available at
#   https://github.com/bearpaw/pytorch-classification
# written by Wei Yang, modified by Linyi Li.

import os
import sys
sys.path.append('.')
sys.path.append('..')
import torch.nn as nn
'''  
python train.py metaroom resnet50 translation_z models/metaroom/resnet50/tz/tv_va --batch 32 --noise_sd 0.0  --lr 0.001  --vanilla --epochs 100

python train.py metaroom resnet18 vanilla models/metaroom/resnet18/tz/tv_va --batch 32 --noise_sd 0.0  --lr 0.001 
python train.py metaroom alexnet translation_z models/metaroom/alexnet/tz/tv_va --batch 32 --noise_sd 0.0  --lr 0.001  --vanilla --epochs 100
python train.py metaroom mnist_43 vanilla models/metaroom/mnist_43/tz/tv_va --batch 32 --noise_sd 0.0  --lr 0.001  --epochs 100
# python train.py metaroom fashion_22full vanilla models/metaroom/fashion_22full/tz/tv_va --batch 32 --noise_sd 0.0  --lr 0.001 
# python train.py metaroom fashion_22simple vanilla models/metaroom/fashion_22simple/tz/tv_va --batch 8 --noise_sd 0.0  --lr 0.001 

python train.py metaroom resnet50 translation_z models/metaroom/resnet50/tz/tv_noise_0.50 --batch 32 --noise_sd 0.5  --lr 0.001 
python train.py metaroom resnet50 translation_z models/metaroom/resnet50/tz/tv_noise_0.0 --batch 32 --noise_sd 0.0  --lr 0.001 
python train.py metaroom resnet18 translation_z models/metaroom/resnet18/tz/tv_noise_0.50 --batch 32 --noise_sd 0.5  --lr 0.001 
python train.py metaroom resnet18 translation_z models/metaroom/resnet18/tz/tv_noise_0.0 --batch 32 --noise_sd 0.0  --lr 0.001 

python train.py metaroom alexnet translation_z models/metaroom/alexnet/tz/tv_noise_0.50 --batch 32 --noise_sd 0.5  --lr 0.001 
python train.py metaroom alexnet translation_z models/metaroom/alexnet/tz/tv_noise_0.0 --batch 32 --noise_sd 0.0  --lr 0.001 
python train.py metaroom mnist_43 translation_z models/metaroom/mnist_43/tz/tv_noise_0.50 --batch 32 --noise_sd 0.5  --lr 0.001 
python train.py metaroom mnist_43 translation_z models/metaroom/mnist_43/tz/tv_noise_0.0 --batch 32 --noise_sd 0.0  --lr 0.001  #--epochs

python train.py metaroom resnet18 translation_z models/metaroom/resnet18/tz/tv_noise_0.1 --batch 32 --noise_sd 0.5  --lr 0.001 
python train.py metaroom resnet18 translation_x models/metaroom/resnet18/tx/tv_noise_0.05 --batch 32   --lr 0.001 
python train.py metaroom resnet18 translation_y models/metaroom/resnet18/ty/tv_noise_0.05 --batch 32   --lr 0.001 
python train.py metaroom resnet18 rotation_z models/metaroom/resnet18/rz/tv_noise_7 --batch 32   --lr 0.001 
python train.py metaroom resnet18 rotation_x models/metaroom/resnet18/rx/tv_noise_2.5 --batch 32   --lr 0.001 
python train.py metaroom resnet18 rotation_y models/metaroom/resnet18/ry/tv_noise_2.5 --batch 32   --lr 0.001 
#python train.py metaroom resnet18 translation_z models/metaroom/resnet18/tz/tv_noise_0.1 --batch 32   --lr 0.001 
# python train.py metaroom resnet18 translation_x models/metaroom/resnet18/tx/tv_noise_0.05 --batch 32   --lr 0.001 
# python train.py metaroom resnet18 translation_y models/metaroom/resnet18/ty/tv_noise_0.05 --batch 32   --lr 0.001 
# python train.py metaroom resnet18 rotation_z models/metaroom/resnet18/rz/tv_noise_7 --batch 32   --lr 0.001 
# python train.py metaroom resnet18 rotation_x models/metaroom/resnet18/rx/tv_noise_2.5 --batch 32   --lr 0.001 
# python train.py metaroom resnet18 rotation_y models/metaroom/resnet18/ry/tv_noise_2.5 --batch 32   --lr 0.001 

#python train.py metaroom resnet18 translation_z models/metaroom/resnet18/tz/tv_va --batch 32 --noise_sd 0.0  --lr 0.001  --vanilla --epochs 100
# python train.py metaroom resnet18 translation_x models/metaroom/resnet18/tx/tv_va --batch 32 --noise_sd 0.0  --lr 0.001  --vanilla --epochs 100
# python train.py metaroom resnet18 translation_y models/metaroom/resnet18/ty/tv_va --batch 32 --noise_sd 0.0  --lr 0.001  --vanilla --epochs 100
# python train.py metaroom resnet18 rotation_x models/metaroom/resnet18/rx/tv_va --batch 32 --noise_sd 0.0  --lr 0.001  --vanilla --epochs 100
# python train.py metaroom resnet18 rotation_y models/metaroom/resnet18/ry/tv_va --batch 32 --noise_sd 0.0  --lr 0.001  --vanilla --epochs 100
# python train.py metaroom resnet18 rotation_z models/metaroom/resnet18/rz/tv_va --batch 32 --noise_sd 0.0  --lr 0.001  --vanilla --epochs 100

# python train.py metaroom resnet18 translation_z models/metaroom/resnet18/tz/tv_noise_0.1_0.5 --batch 32 --noise_sd 0.5   --lr 0.001 
# python train.py metaroom resnet18 translation_x models/metaroom/resnet18/tx/tv_noise_0.05_0.5 --batch 32 --noise_sd 0.5   --lr 0.001 
# python train.py metaroom resnet18 translation_y models/metaroom/resnet18/ty/tv_noise_0.05_0.5 --batch 32 --noise_sd 0.5   --lr 0.001 
# python train.py metaroom resnet18 rotation_z models/metaroom/resnet18/rz/tv_noise_7_0.5 --batch 32  --noise_sd 0.5  --lr 0.001 
# python train.py metaroom resnet18 rotation_x models/metaroom/resnet18/rx/tv_noise_2.5_0.5 --batch 32 --noise_sd 0.5   --lr 0.001 
# python train.py metaroom resnet18 rotation_y models/metaroom/resnet18/ry/tv_noise_2.5_0.5 --batch 32  --noise_sd 0.5  --lr 0.001 

# python train.py metaroom resnet50 translation_y models/metaroom/resnet50/ty/tv_noise_0.05_0.1 --batch 32 --noise_sd 0.1   --lr 0.001 
python train.py metaroom resnet50 rotation_z models/metaroom/resnet50/tz/new_tv_noise_7_0.1 --batch 32  --noise_sd 0.1  --lr 0.001  --pretrain torchvision 

# python train.py metaroom resnet50 rotation_z models/metaroom/resnet50/tz/new_tv_noise_7_0.1 --batch 32  --noise_sd 0.1  --lr 0.001  --pretrained torchvision 


CUDA_VISIBLE_DEVICES=3 python train.py metaroom resnet50 all models/metaroom/resnet50/all/tv_noise_0.0 --batch 128 --noise_sd 0.0   --lr 0.001  --pretrain torchvision 
CUDA_VISIBLE_DEVICES=0 python train.py metaroom resnet50 all models/metaroom/resnet50/all/tv_noise_0.1 --batch 128 --noise_sd 0.1   --lr 0.001  --pretrain torchvision 
CUDA_VISIBLE_DEVICES=2 python train.py metaroom resnet50 all models/metaroom/resnet50/all/noise_0.1 --batch 128 --noise_sd 0.1   --lr 0.001  

CUDA_VISIBLE_DEVICES=1 python train.py metaroom resnet50 all models/metaroom/resnet50/all/tv_noise_0.25 --batch 128 --noise_sd 0.25   --lr 0.001  --pretrain torchvision 
CUDA_VISIBLE_DEVICES=3 python train.py metaroom resnet50 all models/metaroom/resnet50/all/noise_0.25 --batch 128 --noise_sd 0.25   --lr 0.001  

CUDA_VISIBLE_DEVICES=4 python train.py metaroom resnet50 all models/metaroom/resnet50/all/tv_noise_0.5 --batch 128 --noise_sd 0.5   --lr 0.001  --pretrain torchvision  
CUDA_VISIBLE_DEVICES=5 python train.py metaroom resnet50 all models/metaroom/resnet50/all/noise_0.5 --batch 128 --noise_sd 0.5   --lr 0.001 

CUDA_VISIBLE_DEVICES=6 python train.py metaroom resnet50 all models/metaroom/resnet50/all/tv_noise_0.75 --batch 128 --noise_sd 0.75   --lr 0.001  --pretrain torchvision 
CUDA_VISIBLE_DEVICES=7 python train.py metaroom resnet50 all models/metaroom/resnet50/all/noise_0.75 --batch 128 --noise_sd 0.75   --lr 0.001  



CUDA_VISIBLE_DEVICES=0 python train.py metaroom resnet101 all models/metaroom/resnet101/all/tv_noise_0.1 --batch 128 --noise_sd 0.1   --lr 0.001  --pretrain torchvision 
CUDA_VISIBLE_DEVICES=2 python train.py metaroom resnet101 all models/metaroom/resnet101/all/noise_0.1 --batch 128 --noise_sd 0.1   --lr 0.001 

CUDA_VISIBLE_DEVICES=1 python train.py metaroom resnet101 all models/metaroom/resnet101/all/tv_noise_0.25 --batch 128 --noise_sd 0.25   --lr 0.001  --pretrain torchvision 
CUDA_VISIBLE_DEVICES=3 python train.py metaroom resnet101 all models/metaroom/resnet101/all/noise_0.25 --batch 128 --noise_sd 0.25   --lr 0.001 

CUDA_VISIBLE_DEVICES=4 python train.py metaroom resnet101 all models/metaroom/resnet101/all/tv_noise_0.5 --batch 128 --noise_sd 0.5   --lr 0.001  --pretrain torchvision 
CUDA_VISIBLE_DEVICES=5 python train.py metaroom resnet101 all models/metaroom/resnet101/all/noise_0.5 --batch 128 --noise_sd 0.5   --lr 0.001 

CUDA_VISIBLE_DEVICES=6 python train.py metaroom resnet101 all models/metaroom/resnet101/all/tv_noise_0.75 --batch 128 --noise_sd 0.75   --lr 0.001  --pretrain torchvision 
CUDA_VISIBLE_DEVICES=7 python train.py metaroom resnet101 all models/metaroom/resnet101/all/noise_0.75 --batch 128 --noise_sd 0.75   --lr 0.001 

CUDA_VISIBLE_DEVICES=3 python train.py metaroom resnet50 all models/metaroom/resnet50/all_diffusion/tv_noise_0.5 --batch 16 --noise_sd 0.5   --lr 0.001  --pretrain torchvision  --denoiser diffusion &> train_diffusion_0.5.txt

CUDA_VISIBLE_DEVICES=3 python train.py metaroom resnet50 all models/metaroom/resnet50/all_diffusion/tv_noise_0.25 --batch 16 --noise_sd 0.25   --lr 0.001  --pretrain torchvision  --denoiser diffusion --epochs 1
CUDA_VISIBLE_DEVICES=2 python train.py metaroom resnet50 all models/metaroom/resnet50/all_diffusion/tv_noise_0.75 --batch 8 --noise_sd 0.75   --lr 0.001  --pretrain torchvision  --denoiser diffusion --epochs 2

CUDA_VISIBLE_DEVICES=0 python train.py metaroom resnet101 all models/metaroom/resnet101/all_diffusion/tv_noise_0.5 --batch 16 --noise_sd 0.5   --lr 0.001  --pretrain torchvision  --denoiser diffusion 

CUDA_VISIBLE_DEVICES=2 python train.py metaroom resnet50 all models/metaroom/resnet50/all_diffusion/tv_noise_0.25 --batch 16 --noise_sd 0.25   --lr 0.001  --pretrain torchvision  --denoiser diffusion 

'''
import argparse
import torch
import torchvision
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import  DATASETS
from architectures import ARCHITECTURES, get_architecture
from torch.optim import SGD, Optimizer, Adam
from torch.optim.lr_scheduler import StepLR
import time
import datetime
from tensorboardX import SummaryWriter
from train_utils import AverageMeter, accuracy, init_logfile, log
from transformers_ import gen_transformer, AbstractTransformer
from DRM import DiffusionRobustModel
torch.set_num_threads(4)


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('dataset', type=str, choices=DATASETS)
parser.add_argument('arch', type=str)
parser.add_argument('transtype', type=str, help='type of projective transformations',
                    choices=['translation_z', 'translation_x', 'translation_y', 'rotation_z', 'rotation_x',
                             'rotation_y', 'resize', 'gaussian', 'btranslation', 'expgaussian', 'foldgaussian',
                             'rotation-brightness', 'rotation-brightness-contrast', 'resize-brightness',
                             'universal', 'all', 'vnn_tx', 'vnn_ty', 'vnn_tz', 'vnn_rx', 'vnn_ry','vnn_rz'])
# parser.add_argument('transtype', type=str, help='type of semantic transformations',
#                     choices=['rotation-noise', 'noise', 'rotation', 'strict-rotation-noise', 'translation',
#                              'brightness', 'resize', 'gaussian', 'btranslation', 'expgaussian', 'foldgaussian',
#                              'rotation-brightness', 'rotation-brightness-contrast', 'resize-brightness',
#                              'universal'])
parser.add_argument('outdir', type=str, help='folder to save model and training log)')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch', default=256, type=int, metavar='N',
                    help='batchsize (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--lr_step_size', type=int, default=30,
                    help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--noise_sd', default=0.0, type=float,
                    help="standard deviation of Gaussian noise for data augmentation")
parser.add_argument('--rotation_angle', help='constrain the rotation angle to +-rotation angle in degree',
                    type=float, default=180.0)
parser.add_argument('--noise_k', default=0.0, type=float,
                    help="standard deviation of brightness scaling")
parser.add_argument('--noise_b', default=0.0, type=float,
                    help="standard deviation of brightness shift")
parser.add_argument('--blur_lamb', default=0.0, type=float,
                    help="standard deviation of Exponential Gaussian blur, only useful when transtype is universal")
parser.add_argument('--sigma_trans', default=0.0, type=float,
                    help="standard deviation of translation, only useful when transtype is universal")
parser.add_argument('--sl', default=1.0, type=float,
                    help="resize minimum ratio")
parser.add_argument('--sr', default=1.0, type=float,
                    help="resize maximum ratio")
parser.add_argument('--gpu', default=None, type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--print_freq', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--pretrain', default=None, type=str)
##################### arguments for consistency training #####################
parser.add_argument('--num-noise-vec', default=1, type=int,
                    help="number of noise vectors. `m` in the paper.")
parser.add_argument('--lbd', default=0., type=float)
##################### arguments for tensorboard print #####################
parser.add_argument('--print_step', action="store_true")
parser.add_argument('--vanilla', action="store_true")
parser.add_argument('--denoiser', type=str, default='',
                    help='Path to a denoiser to attached before classifier during certificaiton.')
parser.add_argument('--training_type', type=str, default='all',
                    help='all, vnn or separate')
parser.add_argument('--clip', default=1.0, type=float,
                    help="resize maximum ratio")
parser.add_argument('--small', action="store_true")

args = parser.parse_args()

if args.training_type == 'vnn':
    from datasets_vnn import get_dataset, DATASETS, get_normalize_layer
    args.small = True
elif args.small:
    from datasets_vnn import get_dataset, DATASETS, get_normalize_layer
else:
    from datasets import get_dataset, DATASETS, get_normalize_layer

def kl_div(input, targets, reduction='batchmean'):
    return F.kl_div(F.log_softmax(input, dim=1), targets,
                    reduction=reduction)


def _cross_entropy(input, targets, reduction='mean'):
    targets_prob = F.softmax(targets, dim=1)
    xent = (-targets_prob * F.log_softmax(input, dim=1)).sum(1)
    if reduction == 'sum':
        return xent.sum()
    elif reduction == 'mean':
        return xent.mean()
    elif reduction == 'none':
        return xent
    else:
        raise NotImplementedError()

def init_weights(net, init_type='normal', gain=0.02, opt=None):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight'):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.uniform_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)
    net.apply(init_func)

def _entropy(input, reduction='mean'):
    return _cross_entropy(input, input, reduction)

def main():
    if args.gpu != None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    train_dataset = get_dataset(args.dataset, 'train', args.transtype, args.vanilla)
    test_dataset = get_dataset(args.dataset, 'val', args.transtype, args.vanilla)
    pin_memory = (args.dataset == "imagenet") or (args.dataset == "metaroom")
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch,
                              num_workers=args.workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                             num_workers=args.workers, pin_memory=pin_memory)

    model = get_architecture(args.arch, args.dataset)

    if args.pretrain is not None:
        if args.pretrain == 'torchvision':
            # load pretrain model from torchvision
            if args.dataset == 'metaroom':# and args.arch == 'resnet50':
                if args.arch == 'resnet50':
                    if args.training_type != 'vnn':
                        model = torchvision.models.resnet50(False).cuda()
                    else:
                        model = Models['resnet50']()

                    # for name, module in model.named_modules():
                    #     if isinstance(module, torch.nn.MaxPool2d):
                    #         model._modules[name] = torch.nn.MaxPool2d(kernel_size=1, stride=1, padding=0).cuda()
                    # print("###############################3")
                elif args.arch == 'resnet101':
                    if args.training_type != 'vnn':
                        model = torchvision.models.resnet101(False).cuda()
                    else:
                        model = Models['resnet101']()
                    # for name, module in model.named_modules():
                    #     if isinstance(module, torch.nn.MaxPool2d):
                    #         model._modules[name] = torch.nn.MaxPool2d(kernel_size=1, stride=1, padding=0).cuda()
                    # print("###############################3")
                elif args.arch == 'resnet18':
                    if args.training_type != 'vnn':
                        model = torchvision.models.resnet18(False).cuda()
                    else:
                        model = Models['resnet18']()
                    # for name, module in model.named_modules():
                    #     if isinstance(module, torch.nn.MaxPool2d):
                    #         model._modules[name] = torch.nn.MaxPool2d(kernel_size=1, stride=1, padding=0).cuda()
                    # print("###############################3")
                elif args.arch == 'cnn_4layer':
                    model = Models['cnn_4layer'](in_ch=3, in_dim=(32, 56))
                elif args.arch == 'cnn_6layer':
                    model = Models['cnn_6layer'](in_ch=3, in_dim=(32, 56))
                elif args.arch == 'cnn_7layer':
                    model = Models['cnn_7layer'](in_ch=3, in_dim=(32, 56))
                elif args.arch == 'cnn_7layer_bn':
                    model = Models['cnn_7layer_bn'](in_ch=3, in_dim=(32, 56))
                elif args.arch == 'mlp_5layer':
                    model = Models['mlp_5layer'](in_ch=3, in_dim=(32, 56))

                elif args.arch == 'resnet34':
                    if args.training_type != 'vnn':
                        model = torchvision.models.resnet34(False).cuda()
                    else:
                        model = Models['resnet34']()
                # init_weights(model, 'xavier')
                model = model.cuda()
                if args.training_type != 'vnn':
                    normalize_layer = get_normalize_layer(args.dataset).cuda()
                    model = torch.nn.Sequential(normalize_layer, model)


                print(f'loaded from torchvision for imagenet {args.arch}')
            else:
                raise Exception(f'Unsupported pretrain arg {args.pretrain}')
        else:
            # load the base classifier
            if args.arch == 'resnet50':
                if args.training_type != 'vnn':
                    model = torchvision.models.resnet50(False).cuda()
                else:
                    model = Models['resnet50']()

                # for name, module in model.named_modules():
                #     if isinstance(module, torch.nn.MaxPool2d):
                #         model._modules[name] = torch.nn.MaxPool2d(kernel_size=1, stride=1, padding=0).cuda()
                # print("###############################3")
            elif args.arch == 'resnet101':
                if args.training_type != 'vnn':
                    model = torchvision.models.resnet101(False).cuda()
                else:
                    model = Models['resnet101']()
                # for name, module in model.named_modules():
                #     if isinstance(module, torch.nn.MaxPool2d):
                #         model._modules[name] = torch.nn.MaxPool2d(kernel_size=1, stride=1, padding=0).cuda()
                # print("###############################3")
            elif args.arch == 'resnet18':
                if args.training_type != 'vnn':
                    model = torchvision.models.resnet18(False).cuda()
                else:
                    model = Models['resnet18']()
                # for name, module in model.named_modules():
                #     if isinstance(module, torch.nn.MaxPool2d):
                #         model._modules[name] = torch.nn.MaxPool2d(kernel_size=1, stride=1, padding=0).cuda()
                # print("###############################3")
            elif args.arch == 'resnet34':
                if args.training_type != 'vnn':
                    model = torchvision.models.resnet34(False).cuda()
                else:
                    model = Models['resnet34']()
                # for name, module in model.named_modules():
                #     if isinstance(module, torch.nn.MaxPool2d):
                #         model._modules[name] = torch.nn.MaxPool2d(kernel_size=1, stride=1, padding=0).cuda()
                # print("###############################3")
                # fix

            #
            # model = torchvision.models.resnet50(pretrained=False).cuda() if args.arch == 'resnet50' else torchvision.models.resnet101(
            #     pretrained=False).cuda()
            model = model.cuda()
            if args.training_type != 'vnn':
                # fix
                normalize_layer = get_normalize_layer(args.dataset).cuda()
                model = torch.nn.Sequential(normalize_layer, model)
            checkpoint = torch.load(args.pretrain)
            model.load_state_dict(checkpoint['state_dict'])
            print(f'loaded from {args.pretrain}')
    t = -1
    if args.denoiser == 'diffusion':
        model = DiffusionRobustModel(model, train_flag=True, small=args.small)
        # Get the timestep t corresponding to noise level sigma
        target_sigma = args.noise_sd * 2
        real_sigma = 0
        t = 0
        while real_sigma < target_sigma:
            t += 1
            a = model.diffusion.sqrt_alphas_cumprod[t]
            b = model.diffusion.sqrt_one_minus_alphas_cumprod[t]
            real_sigma = b / a
        print("t:", t)



    logfilename = os.path.join(args.outdir, 'log.txt')
    init_logfile(logfilename, "epoch\ttime\tlr\ttrain loss\ttrain acc\ttestloss\ttest acc")
    writer = SummaryWriter(args.outdir)

    canopy = None
    for (inputs, targets) in train_loader:
        canopy = inputs[0]
        break
    transformer = gen_transformer(args, canopy)

    criterion = CrossEntropyLoss().cuda()
    if args.denoiser == 'diffusion':
        optimizer = SGD(model.classifier.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)
    test_acc_best = 0.0
    for epoch in range(args.epochs):
        before = time.time()
        # train_loss, train_acc = 0.0, 0.0
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, transformer, writer, t=t)
        test_loss, test_acc = test(test_loader, model, criterion, epoch, transformer, writer, args.print_freq, t=t)
        after = time.time()

        scheduler.step(epoch)
        try:
            log(logfilename, "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
                epoch, str(datetime.timedelta(seconds=(after - before))),
                scheduler.get_lr()[0], train_loss, train_acc, test_loss, test_acc))
            print("try good", test_acc)
        except:
            log(logfilename, "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
                epoch, str(datetime.timedelta(seconds=(after - before))),
                0, train_loss, train_acc, test_loss, test_acc))
            print("try bad", test_acc)

        if test_acc > test_acc_best:
            torch.save({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'test_acc': test_acc
            }, os.path.join(args.outdir, 'best_checkpoint.pth.tar'))
            test_acc_best = test_acc

        if t < 0:
            torch.save({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.outdir, 'checkpoint.pth.tar'))
        else:
            torch.save({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.outdir, 'checkpoint.pth.tar'))

def _chunk_minibatch(batch, num_batches):
    X, y = batch
    batch_size = len(X) // num_batches
    for i in range(num_batches):
        yield X[i*batch_size : (i+1)*batch_size], y[i*batch_size : (i+1)*batch_size]



def train(loader: DataLoader, model: torch.nn.Module, criterion, optimizer: Optimizer, epoch: int,
          transformer: AbstractTransformer, writer=None, t=-1):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_reg = AverageMeter()
    confidence = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # switch to train mode
    if t < 0:
        model.train()
    else:
        model.train_flag = True
        model.classifier.train()

    for i, batch in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        mini_batches = _chunk_minibatch(batch, args.num_noise_vec)
        for inputs, targets in mini_batches:
            targets = targets.cuda()
            batch_size = inputs.size(0)

            if t < 0 :
                noised_inputs = [transformer.process(inputs).cuda() for _ in range(args.num_noise_vec)]

                # augment inputs with noise
                inputs_c = torch.cat(noised_inputs, dim=0)
            else:
                inputs_c = inputs.cuda()
                # print(args.num_noise_vec, inputs.shape)
                # inputs_c = inputs.repeat(args.num_noise_vec)
            targets_c = targets.repeat(args.num_noise_vec)
            # print(targets_c.shape)

            if t < 0:
                logits = model(inputs_c)
            else:
                if args.small:
                    inputs_c = torch.nn.functional.interpolate(inputs_c, 32)
                else:
                    inputs_c = torch.nn.functional.interpolate(inputs_c, 256)
                logits = model(inputs_c, t)

            loss_xent = criterion(logits, targets_c)

            logits_chunk = torch.chunk(logits, args.num_noise_vec, dim=0)
            softmax = [F.softmax(logit, dim=1) for logit in logits_chunk]
            avg_softmax = sum(softmax) / args.num_noise_vec

            consistency = [kl_div(logit, avg_softmax, reduction='none').sum(1)
                           + _entropy(avg_softmax, reduction='none')
                           for logit in logits_chunk]
            consistency = sum(consistency) / args.num_noise_vec
            consistency = consistency.mean()

            loss = loss_xent #+ args.lbd * consistency

            avg_confidence = -F.nll_loss(avg_softmax, targets)

            acc1, acc5 = accuracy(logits, targets_c, topk=(1, 5))
            losses.update(loss_xent.item(), batch_size)
            losses_reg.update(consistency.item(), batch_size)
            confidence.update(avg_confidence.item(), batch_size)
            top1.update(acc1.item(), batch_size)
            top5.update(acc5.item(), batch_size)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            # print(loss, loss_xent)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'Loss {loss.avg:.4f}\t'
                  'Acc@1 {top1.avg:.3f}\t'
                  'Acc@5 {top5.avg:.3f}'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

            if args.print_step:
                writer.add_scalar(f'epoch/{epoch}/loss/train', losses.avg, i)
                writer.add_scalar(f'epoch/{epoch}/loss/consistency', losses_reg.avg, i)
                writer.add_scalar(f'epoch/{epoch}/loss/avg_confidence', confidence.avg, i)
                writer.add_scalar(f'epoch/{epoch}/batch_time', batch_time.avg, i)
                writer.add_scalar(f'epoch/{epoch}/accuracy/train@1', top1.avg, i)
                writer.add_scalar(f'epoch/{epoch}/accuracy/train@5', top5.avg, i)

    writer.add_scalar('loss/train', losses.avg, epoch)
    writer.add_scalar('loss/consistency', losses_reg.avg, epoch)
    writer.add_scalar('loss/avg_confidence', confidence.avg, epoch)
    writer.add_scalar('batch_time', batch_time.avg, epoch)
    writer.add_scalar('accuracy/train@1', top1.avg, epoch)
    writer.add_scalar('accuracy/train@5', top5.avg, epoch)

    return (losses.avg, top1.avg)


def test(loader, model, criterion, epoch, transformer: AbstractTransformer, writer=None, print_freq=10, t=-1):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # switch to eval mode
    if t < 0:
        model.eval()
    else:
        model.classifier.eval()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs = inputs
            targets = targets.cuda()

            if t < 0:
                # augment inputs with noise
                inputs = transformer.process(inputs).cuda()
                # compute output
                outputs = model(inputs)
            else:
                model.train_flag = False
                inputs = inputs.cuda()
                if args.small:
                    inputs = torch.nn.functional.interpolate(inputs, 32)
                else:
                    inputs = torch.nn.functional.interpolate(inputs, 256)
                # compute output
                outputs = model(inputs, t)

            # compute output
            # outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.avg:.3f}\t'
                      'Data {data_time.avg:.3f}\t'
                      'Loss {loss.avg:.4f}\t'
                      'Acc@1 {top1.avg:.3f}\t'
                      'Acc@5 {top5.avg:.3f}'.format(
                    i, len(loader), batch_time=batch_time, data_time=data_time,
                    loss=losses, top1=top1, top5=top5))
                # if args.print_step:
                #     writer.add_scalar(f'epoch/{epoch}/test/loss', losses.avg, i)
                #     writer.add_scalar(f'epoch/{epoch}/test/train@1', top1.avg, i)
                #     writer.add_scalar(f'epoch/{epoch}/test/train@5', top5.avg, i)

        # if writer:
        writer.add_scalar('loss/test', losses.avg, epoch)
        writer.add_scalar('accuracy/test@1', top1.avg, epoch)
        writer.add_scalar('accuracy/test@5', top5.avg, epoch)

        return (losses.avg, top1.avg)

# def train(loader: DataLoader, model: torch.nn.Module, criterion, optimizer: Optimizer, epoch: int, transformer: AbstractTransformer):
#     batch_time = AverageMeter()
#     data_time = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()
#     top5 = AverageMeter()
#     end = time.time()
#
#     # switch to train mode
#     model.train()
#
#     for i, (inputs, targets) in enumerate(loader):
#         # measure data loading time
#         data_time.update(time.time() - end)
#
#         inputs = inputs
#         targets = targets.cuda()
#
#         # augment inputs with noise
#         inputs = transformer.process(inputs).cuda()
#
#         # compute output
#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
#
#         # measure accuracy and record loss
#         acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
#         losses.update(loss.item(), inputs.size(0))
#         top1.update(acc1.item(), inputs.size(0))
#         top5.update(acc5.item(), inputs.size(0))
#
#         # compute gradient and do SGD step
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()
#
#         if i % args.print_freq == 0:
#             print('Epoch: [{0}][{1}/{2}]\t'
#                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
#                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                   'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
#                   'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
#                 epoch, i, len(loader), batch_time=batch_time,
#                 data_time=data_time, loss=losses, top1=top1, top5=top5))
#
#     return (losses.avg, top1.avg)
#
#
# def test(loader: DataLoader, model: torch.nn.Module, criterion, transformer: AbstractTransformer):
#     batch_time = AverageMeter()
#     data_time = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()
#     top5 = AverageMeter()
#     end = time.time()
#
#     # switch to eval mode
#     model.eval()
#
#     with torch.no_grad():
#         for i, (inputs, targets) in enumerate(loader):
#             # measure data loading time
#             data_time.update(time.time() - end)
#
#             inputs = inputs
#             targets = targets.cuda()
#
#             # augment inputs with noise
#             inputs = transformer.process(inputs).cuda()
#
#             # compute output
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#
#             # measure accuracy and record loss
#             acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
#             losses.update(loss.item(), inputs.size(0))
#             top1.update(acc1.item(), inputs.size(0))
#             top5.update(acc5.item(), inputs.size(0))
#
#             # measure elapsed time
#             batch_time.update(time.time() - end)
#             end = time.time()
#
#             if i % args.print_freq == 0:
#                 print('Test: [{0}/{1}]\t'
#                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                       'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
#                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                       'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
#                       'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
#                     i, len(loader), batch_time=batch_time,
#                     data_time=data_time, loss=losses, top1=top1, top5=top5))
#
#         return (losses.avg, top1.avg)


if __name__ == "__main__":
    main()
