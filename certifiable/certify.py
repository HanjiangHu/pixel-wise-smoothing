
import os
import sys
sys.path.append('.')
sys.path.append('..')

# evaluate a smoothed classifier on a dataset 
import argparse
# import setGPU
from tensorboardX import SummaryWriter
from datasets import get_dataset, DATASETS, get_num_classes, get_normalize_layer
from core import SemanticSmooth
from time import time
import torch
import torchvision
import datetime
from architectures import get_architecture
from transformers_ import gen_transformer
from torch.utils.data import DataLoader

'''
python certify.py metaroom models/metaroom/resnet50/tz/tv_noise_0.1/checkpoint.pth.tar 0.1 resolvable_tz data/predict/metaroom/resnet50/tz/tv_noise_0.1 --batch 200
CUDA_VISIBLE_DEVICES=1 python certify.py metaroom models/metaroom/resnet18/tz/tv_noise_0.1/checkpoint.pth.tar 0.1 resolvable_tz data/predict/metaroom/resnet18/tz/tv_noise_0.1 --batch 200

python certify.py metaroom models/metaroom/resnet18/ty/tv_noise_0.05/checkpoint.pth.tar 0.05 resolvable_ty data/predict/metaroom/resnet18/ty/tv_noise_0.05 --batch 200

CUDA_VISIBLE_DEVICES=1 python certify.py metaroom models/metaroom/resnet18/tx/tv_noise_0.05/checkpoint.pth.tar 0.05 resolvable_tx data/predict/metaroom/resnet18/tx/tv_noise_0.05 --batch 200

python certify.py metaroom models/metaroom/resnet18/rx/tv_noise_2.5/checkpoint.pth.tar 0.04363323 resolvable_rx data/predict/metaroom/resnet18/rx/tv_noise_2.5 --batch 200
python certify.py metaroom models/metaroom/resnet18/ry/tv_noise_2.5/checkpoint.pth.tar 0.04363323 resolvable_ry data/predict/metaroom/resnet18/ry/tv_noise_2.5 --batch 200
python certify.py metaroom models/metaroom/resnet18/rz/tv_noise_7/checkpoint.pth.tar 0.122173 resolvable_rz data/predict/metaroom/resnet18/rz/tv_noise_7 --batch 200

'''

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("noise_sd", type=float, help="noise hyperparameter")
parser.add_argument('transtype', type=str, help='type of projective transformations',
                    choices=['resolvable_tz', 'resolvable_tx', 'resolvable_ty', 'resolvable_rz', 'resolvable_rx',
                             'resolvable_ry', 'diff_resolvable_tz', 'diff_resolvable_tx', 'diff_resolvable_ty', 'diff_resolvable_rz', 'diff_resolvable_rx',
                             'diff_resolvable_ry'])
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument('--noise_k', default=0.0, type=float,
                    help="standard deviation of brightness scaling")
parser.add_argument('--noise_b', default=0.0, type=float,
                    help="standard deviation of brightness shift")
parser.add_argument("--bright_scale", type=float, default=0.0,
                    help="for brightness transformation, the scale interval is 1.0 +- bright_scale")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--start", type=int, default=0, help="start before skipping how many examples")
parser.add_argument("--split", default="certify", help="images to certify, the same images as test (benign)")
parser.add_argument("--N0", type=int, default=500) #100
parser.add_argument("--N", type=int, default=10000, help="number of samples to use") # 1000
parser.add_argument("--alpha", type=float, default=0.01, help="failure probability")
parser.add_argument('--gpu', default=None, type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument("--th", type=float, default=0, help="pre-defined radius for true robust counting")
parser.add_argument("--saved_path", type=str, default=None, help='output of alias data')
parser.add_argument("--using_save", action='store_true')
parser.add_argument("--sigma", type=float, default=-1, help="real sigma is */3500")
parser.add_argument("--vnn", action='store_true')
# parser.add_argument("--uniform_smth", action='store_true')
args = parser.parse_args()

if __name__ == "__main__":

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # # load the base classifier
    # checkpoint = torch.load(args.base_classifier)
    # base_classifier = get_architecture(checkpoint["arch"], args.dataset)
    # print('arch:', checkpoint['arch'])
    # if checkpoint["arch"] == 'resnet50' and args.dataset == "imagenet":
    #     try:
    #         base_classifier.load_state_dict(checkpoint['state_dict'])
    #     except Exception as e:
    #         print('direct load failed, try alternative')
    #         try:
    #             base_classifier = torchvision.models.resnet50(pretrained=False).cuda()
    #             base_classifier.load_state_dict(checkpoint['state_dict'])
    #             # fix
    #             # normalize_layer = get_normalize_layer('imagenet').cuda()
    #             # base_classifier = torch.nn.Sequential(normalize_layer, base_classifier)
    #         except Exception as e:
    #             print('alternative failed again, try alternative 2')
    #             base_classifier = torchvision.models.resnet50(pretrained=False).cuda()
    #             # base_classifier.load_state_dict(checkpoint['state_dict'])
    #             normalize_layer = get_normalize_layer('imagenet').cuda()
    #             base_classifier = torch.nn.Sequential(normalize_layer, base_classifier)
    #             base_classifier.load_state_dict(checkpoint['state_dict'])
    # else:
    #     print("#######################################")
    #     base_classifier.load_state_dict(checkpoint['state_dict'])
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.vnn:
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
        base_classifier.load_state_dict(checkpoint['state_dict'])
        base_classifier = base_classifier.cuda()
        print(f"loaded {checkpoint['arch']}")
    else:

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


    # normalize_layer = get_normalize_layer(args.dataset).cuda()
    # base_classifier = torch.nn.Sequential(normalize_layer, base_classifier)
    # assert 1 == 2


    # prepare output file
    if not os.path.exists(os.path.dirname(args.outfile)):
        os.makedirs(os.path.dirname(args.outfile))
    f = open(args.outfile, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split, args.transtype)
    # test_loader = DataLoader(dataset, shuffle=False, batch_size=1,
    #                          num_workers=args.workers, pin_memory=True)
    if args.sigma == -1:
        transformer = gen_transformer(args, dataset[0][0], vnn=args.vnn, uniform_smth=True)
    else:
        transformer = gen_transformer(args, dataset[0][0], vnn=args.vnn)

    # special setting for brightness
    if args.transtype == 'brightness':
        transformer.set_brightness_scale(1.0 - args.bright_scale, 1.0 + args.bright_scale)
    if args.transtype == 'contrast':
        # binary search from 0.1 to 10.0
        transformer.set_contrast_scale(0.1, 10.0)

    # init tensorboard writer
    writer = SummaryWriter(os.path.dirname(args.outfile))

    # create the smooothed classifier g
    smoothed_classifier = SemanticSmooth(base_classifier, get_num_classes(args.dataset), transformer)

    tot_clean, tot_good, tot = 0, 0, 0

    for i in range(len(dataset)):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break
        if i < args.start:
            continue

        (x, label) = dataset[i]

        before_time = time()
        # certify the prediction of g around x
        # x = x.cuda()
        prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch, indx=i, save_flag=args.using_save, save_path=args.saved_path, sigma=args.sigma)
        after_time = time()
        correct = int(prediction == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)
        print(i, time_elapsed, correct, radius)

        tot += 1
        tot_clean += correct
        tot_good += int(radius > args.th if correct > 0 else 0)
        writer.add_scalar('certify/clean_acc', tot_clean / tot, i)
        # writer.add_scalar('certify/robust_acc', tot_cert / tot, i)
        writer.add_scalar('certify/true_robust_acc', tot_good / tot, i)

    f.close()
