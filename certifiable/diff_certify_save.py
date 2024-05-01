
import os
import sys
sys.path.append('.')
sys.path.append('..')

import math

# evaluate a smoothed classifier on a dataset
import argparse
# import setGPU
from datasets import get_dataset, DATASETS, get_num_classes, get_normalize_layer
from core import SemanticSmooth
from time import time
from DRM import DiffusionRobustModel
import random
# import setproctitle
import torch
import torchvision
import datetime
from tensorboardX import SummaryWriter

from architectures import get_architecture
from architectures_denoise import get_architecture_denoise
from transformers_ import RotationTransformer
from transformers_ import gen_transformer, DiffResolvableProjectionTransformer
from transforms import visualize
import cupy as np
import numpy
import matplotlib
import matplotlib.pyplot as plt
from collections import OrderedDict

'''
python diff_certify.py metaroom diff_resolvable_tz models/metaroom/resnet18/tz/tv_noise_0.1_0.1/checkpoint.pth.tar 0.1 data/alias/metaroom/tz/tv_noise_0.1_0.1 data/predict/metaroom/resnet18/tz/tv_noise_0.1_0.1  --batch 200  --partial 0.1
'''

EPS = 1e-6

parser = argparse.ArgumentParser(description='Strict rotation certify')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument('transtype', type=str, help='type of projective transformations',
                    choices=['resolvable_tz', 'resolvable_tx', 'resolvable_ty', 'resolvable_rz', 'resolvable_rx',
                             'resolvable_ry', 'diff_resolvable_tz', 'diff_resolvable_tx', 'diff_resolvable_ty', 'diff_resolvable_rz', 'diff_resolvable_rx',
                             'diff_resolvable_ry'])
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")

parser.add_argument("noise_sd", type=float, help="pixel gaussian noise hyperparameter")
parser.add_argument("--noise_b", type=float, default=0.0, help="noise hyperparameter for brightness shift dimension")
parser.add_argument("--noise_k", type=float, default=0.0, help="noise hyperparameter for brightness scaling dimension")
parser.add_argument("--l2_r", type=float, default=0.0, help="additional l2 magnitude to be tolerated")
parser.add_argument("aliasfile", type=str, help='output of alias data')
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--b", type=float, default=0.0, help="brightness shift requirement")
parser.add_argument("--k", type=float, default=0.0, help="brightness scaling requirement")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--start", type=int, default=0, help="start before skipping how many examples")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", default="certify", help="train or test set")
parser.add_argument("--N0", type=int, default=500)
parser.add_argument("--N", type=int, default=10000, help="number of samples to use")
parser.add_argument("--slice", type=int, default=1000, help="number of angle slices")
parser.add_argument("--alpha", type=float, default=0.01, help="failure probability")
parser.add_argument("--partial", type=float, default=180.0, help="certify +-partial degrees")
parser.add_argument("--verbstep", type=int, default=10, help="output frequency")
parser.add_argument('--gpu', default=None, type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument("--N_partitions", type=int, default=7001, help="number of partitions to use")
parser.add_argument("--saved_path", type=str, default="./data/alias_save/metaroom/tx/tv_noise_0.5_7000_imgs_fixed", help='output of alias data')
parser.add_argument("--factor", type=float, default=1.0, help="factors to rescale from original radii")
parser.add_argument('--denoiser', type=str, default='',
                    help='Path to a denoiser to attached before classifier during certificaiton.')
parser.add_argument('--training_type', type=str, default='all',
                    help='all or separate or vnn')
parser.add_argument('--small', action="store_true")

args = parser.parse_args()

if args.training_type == 'vnn':
    from datasets_vnn import get_dataset, DATASETS, get_normalize_layer
    args.small = True
elif args.small:
    from datasets_vnn import get_dataset, DATASETS, get_normalize_layer
else:
    from datasets import get_dataset, DATASETS, get_normalize_layer

def DataParallel2CPU(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:9] == "1.module.":
            k = "1." + k[9:]
        new_state_dict[k] = v
    return new_state_dict

if __name__ == '__main__':
    orig_alpha = args.alpha
    args.alpha /= (args.slice)# * (2.0 * args.partial) / 360.0 + 1)
    t = -1

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


    if args.training_type == 'all':
        # load the base classifier
        checkpoint = torch.load(args.base_classifier)
        base_classifier = get_architecture(checkpoint["arch"], args.dataset)
        if checkpoint["arch"] == 'resnet50' or checkpoint["arch"] == 'resnet101' or checkpoint["arch"] == 'resnet18':
            # assert 1==2
            try:
                base_classifier.load_state_dict(checkpoint['state_dict'])
            except:
                if checkpoint["arch"] == 'resnet50':
                    base_classifier = torchvision.models.resnet50(pretrained=False).cuda()
                elif checkpoint["arch"] == 'resnet101':
                    base_classifier = torchvision.models.resnet101(pretrained=False).cuda()
                else:
                    base_classifier = torchvision.models.resnet18(pretrained=False).cuda()
                # fix
                normalize_layer = get_normalize_layer(args.dataset).cuda()
                base_classifier = torch.nn.Sequential(normalize_layer, base_classifier)
                print("$$$$$$$$$$$$")
        base_classifier.load_state_dict(checkpoint['state_dict'])
        if args.denoiser != '':
            if args.denoiser == 'diffusion':
                base_classifier = DiffusionRobustModel(base_classifier, small=args.small)
                # Get the timestep t corresponding to noise level sigma
                target_sigma = args.noise_sd * 2
                real_sigma = 0
                t = 0
                while real_sigma < target_sigma:
                    t += 1
                    a = base_classifier.diffusion.sqrt_alphas_cumprod[t]
                    b = base_classifier.diffusion.sqrt_one_minus_alphas_cumprod[t]
                    real_sigma = b / a
                print("t:", t)

            else:
                checkpoint_denoiser = torch.load(args.denoiser)
                if "off-the-shelf-denoiser" in args.denoiser:
                    denoiser = get_architecture_denoise('orig_dncnn', args.dataset)
                    denoiser.load_state_dict(checkpoint_denoiser)
                else:
                    denoiser = get_architecture_denoise(checkpoint_denoiser['arch'], args.dataset)
                    denoiser.load_state_dict(checkpoint_denoiser['state_dict'])
                base_classifier = torch.nn.Sequential(denoiser, base_classifier)
                print("denoiser added")
    elif args.training_type == 'vnn':
        checkpoint = torch.load(args.base_classifier)

        # if checkpoint["arch"] == 'resnet50':
        #     base_classifier = torchvision.models.resnet50(False)
        #     for name, module in base_classifier.named_modules():
        #         if isinstance(module, torch.nn.MaxPool2d):
        #             base_classifier._modules[name] = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        #     print("###############################3")
        if checkpoint["arch"] == 'resnet50':
            # if args.training_type != 'vnn':
            #     base_classifier = torchvision.models.resnet50(False).cuda()
            # else:
            base_classifier = Models['resnet50']()

            # for name, module in model.named_modules():
            #     if isinstance(module, torch.nn.MaxPool2d):
            #         model._modules[name] = torch.nn.MaxPool2d(kernel_size=1, stride=1, padding=0).cuda()
            print("###############################3")
        elif checkpoint["arch"] == 'resnet101':
            # if args.training_type != 'vnn':
            #     base_classifier = torchvision.models.resnet101(False).cuda()
            # else:
            base_classifier = Models['resnet101']()
            # for name, module in model.named_modules():
            #     if isinstance(module, torch.nn.MaxPool2d):
            #         model._modules[name] = torch.nn.MaxPool2d(kernel_size=1, stride=1, padding=0).cuda()
            print("###############################3")
        elif checkpoint["arch"] == 'resnet18':
            # if args.training_type != 'vnn':
            #     base_classifier = torchvision.models.resnet18(False).cuda()
            # else:
            base_classifier = Models['resnet18']()
            # for name, module in model.named_modules():
            #     if isinstance(module, torch.nn.MaxPool2d):
            #         model._modules[name] = torch.nn.MaxPool2d(kernel_size=1, stride=1, padding=0).cuda()
            print("###############################3")
        elif checkpoint["arch"] == 'resnet34':
            # if args.training_type != 'vnn':
            #     base_classifier = torchvision.models.resnet34(False).cuda()
            # else:
            base_classifier = Models['resnet34']()
        elif checkpoint["arch"] == 'cnn_4layer':
            base_classifier = Models['cnn_4layer'](in_ch=3, in_dim=(32, 56))
        elif checkpoint["arch"] == 'cnn_6layer':
            base_classifier = Models['cnn_6layer'](in_ch=3, in_dim=(32, 56))
        elif checkpoint["arch"] == 'cnn_7layer':
            base_classifier = Models['cnn_7layer'](in_ch=3, in_dim=(32, 56))
        elif checkpoint["arch"] == 'cnn_7layer_bn':
            base_classifier = Models['cnn_7layer_bn'](in_ch=3, in_dim=(32, 56))

        elif checkpoint["arch"] == 'mlp_5layer':
            base_classifier = Models['mlp_5layer'](in_ch=3, in_dim=(32, 56))
        else:
            base_classifier = None
            print('not supported')
        state_dict = DataParallel2CPU(checkpoint['state_dict'])
        base_classifier.load_state_dict(state_dict)
        base_classifier = base_classifier.cuda()
        print(f"loaded {checkpoint['arch']}")
        if args.denoiser != '':

            if args.denoiser == 'diffusion':
                base_classifier = DiffusionRobustModel(base_classifier, small=args.small)
                # Get the timestep t corresponding to noise level sigma
                target_sigma = args.noise_sd * 2
                real_sigma = 0
                t = 0
                while real_sigma < target_sigma:
                    t += 1
                    a = base_classifier.diffusion.sqrt_alphas_cumprod[t]
                    b = base_classifier.diffusion.sqrt_one_minus_alphas_cumprod[t]
                    real_sigma = b / a

            else:
                checkpoint_denoiser = torch.load(args.denoiser)
                if "off-the-shelf-denoiser" in args.denoiser:
                    denoiser = get_architecture_denoise('orig_dncnn', args.dataset)
                    denoiser.load_state_dict(checkpoint_denoiser)
                else:
                    denoiser = get_architecture_denoise(checkpoint_denoiser['arch'], args.dataset)
                    denoiser.load_state_dict(checkpoint_denoiser['state_dict'])
                base_classifier = torch.nn.Sequential(denoiser, base_classifier)
                print("denoiser added")
    else:
        # load the base classifier
        checkpoint = torch.load(args.base_classifier)
        base_classifier = get_architecture(checkpoint["arch"], args.dataset)
        print('arch:', checkpoint['arch'])

        if checkpoint["arch"] == 'resnet50' and args.dataset == "imagenet":
            try:
                base_classifier.load_state_dict(checkpoint['state_dict'])
            except Exception as e:
                print('direct load failed, try alternative')
                try:
                    base_classifier = torchvision.models.resnet50(pretrained=False).cuda()
                    base_classifier.load_state_dict(checkpoint['state_dict'])
                    # fix
                    # normalize_layer = get_normalize_layer('imagenet').cuda()
                    # base_classifier = torch.nn.Sequential(normalize_layer, base_classifier)
                except Exception as e:
                    print('alternative failed again, try alternative 2')
                    base_classifier = torchvision.models.resnet50(pretrained=False).cuda()
                    # base_classifier.load_state_dict(checkpoint['state_dict'])
                    normalize_layer = get_normalize_layer('imagenet').cuda()
                    base_classifier = torch.nn.Sequential(normalize_layer, base_classifier)
                    base_classifier.load_state_dict(checkpoint['state_dict'])
        else:
            print("#######################################")
            base_classifier.load_state_dict(checkpoint['state_dict'])
        if args.denoiser != '':

            if args.denoiser == 'diffusion':
                base_classifier = DiffusionRobustModel(base_classifier, small=args.small)
                # Get the timestep t corresponding to noise level sigma
                target_sigma = args.noise_sd * 2
                real_sigma = 0
                t = 0
                while real_sigma < target_sigma:
                    t += 1
                    a = base_classifier.diffusion.sqrt_alphas_cumprod[t]
                    b = base_classifier.diffusion.sqrt_one_minus_alphas_cumprod[t]
                    real_sigma = b / a

            else:
                checkpoint_denoiser = torch.load(args.denoiser)
                if "off-the-shelf-denoiser" in args.denoiser:
                    denoiser = get_architecture_denoise('orig_dncnn', args.dataset)
                    denoiser.load_state_dict(checkpoint_denoiser)
                else:
                    denoiser = get_architecture_denoise(checkpoint_denoiser['arch'], args.dataset)
                    denoiser.load_state_dict(checkpoint_denoiser['state_dict'])
                base_classifier = torch.nn.Sequential(denoiser, base_classifier)
                print("denoiser added")

        # else:
        #     base_classifier.load_state_dict(checkpoint['state_dict'])

    # normalize_layer = get_normalize_layer(args.dataset).cuda()
    # base_classifier = torch.nn.Sequential(normalize_layer, base_classifier)
    # assert 1 == 2
    # base_classifier.load_state_dict(checkpoint['state_dict'])

    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split, args.transtype)

    # init transformers
    # rotationT = RotationTransformer(dataset[0][0])
    diff_projection = DiffResolvableProjectionTransformer(dataset[0][0], args.transtype[-2:])

    # build Gaussian smoothing on pixel
    transformer = gen_transformer(args, dataset[0][0])
    # transformer = None
    # if abs(args.noise_b) < EPS and abs(args.noise_k) < EPS:
    #     transformer = GaussianTransformer(args.noise_sd)
    # if abs(args.noise_k) < EPS:
    #     transformer = RotationBrightnessNoiseTransformer(args.noise_sd, args.noise_b, dataset[0][0], 0.)
    #     transformer.set_brightness_shift(args.b)
    #     transformer.rotation_adder.mask = transformer.rotation_adder.mask.cuda()
    # else:
    #     transformer = RotationBrightnessContrastNoiseTransformer(args.noise_sd, args.noise_b, args.noise_k, dataset[0][0], 0.)
    #     transformer.set_brightness_shift(args.b)
    #     transformer.set_brightness_scale(1.0 - args.k, 1.0 + args.k)
    #     transformer.rotation_adder.mask = transformer.rotation_adder.mask.cuda()

    # init alias analysis
    alias_dic = dict()
    if args.N_partitions == 7001:
        f = open(args.aliasfile, 'r')
        for line in f.readlines()[1:]:
            try:
                no, v, num_slice = line.split('\t')
            except:
                # sometimes we have the last column for time recording
                no, v, num_slice, _ = line.split('\t')
            no, v, num_slice = int(no), float(v), int(num_slice)
            alias_dic[no] = [v, num_slice]
    else:
        for no in range(120):
            path = args.saved_path + '/%03d' % no
            image_list = []
            for indx in range(args.N_partitions):
                inx = int(7001 * indx / args.N_partitions)
                image_i = matplotlib.image.imread(path+ '/%05d.png' % inx)
                image_list.append(image_i.reshape(-1))
            image_all = np.asarray(numpy.stack(image_list))
            M = np.amax(np.sqrt(np.sum(np.square(image_all[:-1, :] - image_all[1:, :]), axis=1) / 2), axis=0)
            alias_dic[no] = [M, args.N_partitions]


    # modify outfile name to distinguish different parts
    # if args.start != 0 or args.max != -1:
    #     args.outfile += f'_start_{args.start}_end_{args.max}'

    # setproctitle.setproctitle(f'rotation_certify_{args.dataset}from{args.start}to{args.max}')

    # prepare output file
    if not os.path.exists(os.path.dirname(args.outfile)):
        os.makedirs(os.path.dirname(args.outfile))
    f = open(args.outfile, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    # init tensorboard writer
    writer = SummaryWriter(os.path.dirname(args.outfile))

    # create the smooothed classifier g
    base_classifier = base_classifier.cuda()
    smoothed_classifier = SemanticSmooth(base_classifier, get_num_classes(args.dataset), transformer, diff=True, t=t, small=args.small)

    tot, tot_clean, tot_good, tot_cert = 0, 0, 0, 0

    for i in range(len(dataset)):
        # print(len(dataset))

        if i < args.start:
            continue

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i >= args.max >= 0:
            break

        (x, label) = dataset[i]
        if i not in alias_dic:
            continue

        margin = alias_dic[i]
        margin[0] = (math.sqrt(margin[0]) + args.l2_r) ** 2
        print('working on #', i, 'max aliasing:', alias_dic[i][0], '->', margin, 'with slices of', margin[1])

        before_time = time()
        image_i = matplotlib.image.imread(args.saved_path + '/%03d' % i + '/%05d.png' % 3500)
        img = torch.as_tensor(image_i[:, :, :3], device=torch.device('cuda'))
        # img2 = diff_projection.projection_adder.proc(x, 0).cuda()
        cAHat = smoothed_classifier.predict(img, args.N0, orig_alpha, args.batch)

        clean, cert, good = (cAHat == label), True, True
        gap = -1.0

        for j in range(margin[1]):
            # if min(360.0 * j / args.slice, 360.0 - 360.0 * (j + 1) / args.slice) >= args.partial:
            #     continue
            if j % args.verbstep == 0:
                print(f"> {j}/{margin[1]} {str(datetime.timedelta(seconds=(time() - before_time)))}", end='\r', flush=True)

            image_i_j = matplotlib.image.imread(args.saved_path + '/%03d' % i + '/%05d.png' % int(7001 * (3500 + (j - 3500) * args.factor) / margin[1]))
            now_img = torch.as_tensor(image_i_j[:, :, :3], device=torch.device('cuda'))

            # now_img = diff_projection.projection_adder.proc(x, -args.partial + 2 * args.partial * j / margin[1]).cuda()
            prediction, gap = smoothed_classifier.certify(now_img, args.N0, args.N, args.alpha, args.batch,
                                                          cAHat=cAHat, margin=margin[0])
            if prediction != cAHat or gap < 0 or cAHat == smoothed_classifier.ABSTAIN:
                print(prediction)
                print(cAHat)
                print(gap)
                print(f'not robust @ slice #{j}')
                good = cert = False
                break
            elif prediction != label:
                # the prediction is robustly wrong:
                print(f'wrong @ slice #{j}')
                # make gap always smaller than 0 for wrong slice
                gap = - abs(gap) - 1.0
                good = False
                # robustly wrong is also skipped
                # now "cert" is not recorded anymore
                break
            # else it is good


        after_time = time()
        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            i, label, cAHat, gap, clean, time_elapsed), file=f, flush=True)

        tot, tot_clean, tot_cert, tot_good = tot + 1, tot_clean + int(clean), tot_cert + int(cert), tot_good + int(good)
        print(f'{i} {gap >= 0.0} '
              f'CleanACC = {tot_clean}/{tot} = {float(tot_clean) / float(tot)} '
              # f'CertAcc = {tot_cert}/{tot} = {float(tot_cert) / float(tot)} '
              f'RACC = {tot_good}/{tot} = {float(tot_good) / float(tot)} '
              f'Time = {time_elapsed}')

        writer.add_scalar('certify/clean_acc', tot_clean / tot, i)
        # writer.add_scalar('certify/robust_acc', tot_cert / tot, i)
        writer.add_scalar('certify/true_robust_acc', tot_good / tot, i)

    print(f'CleanACC = {tot_clean}/{tot} = {float(tot_clean) / float(tot)} '
        # f'CertAcc = {tot_cert}/{tot} = {float(tot_cert) / float(tot)} '
        f'RACC = {tot_good}/{tot} = {float(tot_good) / float(tot)}', file=f, flush=True)

    f.close()




