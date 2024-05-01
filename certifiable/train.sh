CUDA_VISIBLE_DEVICES=0 python train.py metaroom resnet50 all models/metaroom/resnet50/all/tv_noise_0.1 --batch 128 --noise_sd 0.1   --lr 0.001  --pretrain torchvision
#CUDA_VISIBLE_DEVICES=0 python train.py metaroom resnet50 all models/metaroom/resnet50/all/noise_0.1 --batch 128 --noise_sd 0.1   --lr 0.001

CUDA_VISIBLE_DEVICES=0 python train.py metaroom resnet50 all models/metaroom/resnet50/all/tv_noise_0.25 --batch 128 --noise_sd 0.25   --lr 0.001  --pretrain torchvision
#CUDA_VISIBLE_DEVICES=0 python train.py metaroom resnet50 all models/metaroom/resnet50/all/noise_0.25 --batch 128 --noise_sd 0.25   --lr 0.001

CUDA_VISIBLE_DEVICES=0 python train.py metaroom resnet50 all models/metaroom/resnet50/all/tv_noise_0.5 --batch 128 --noise_sd 0.5   --lr 0.001  --pretrain torchvision
#CUDA_VISIBLE_DEVICES=0 python train.py metaroom resnet50 all models/metaroom/resnet50/all/noise_0.5 --batch 128 --noise_sd 0.5   --lr 0.001

CUDA_VISIBLE_DEVICES=0 python train.py metaroom resnet50 all models/metaroom/resnet50/all/tv_noise_0.75 --batch 128 --noise_sd 0.75   --lr 0.001  --pretrain torchvision
#CUDA_VISIBLE_DEVICES=0 python  train.py metaroom resnet50 all models/metaroom/resnet50/all/noise_0.75 --batch 128 --noise_sd 0.75   --lr 0.001



CUDA_VISIBLE_DEVICES=0 python train.py metaroom resnet101 all models/metaroom/resnet101/all/tv_noise_0.1 --batch 128 --noise_sd 0.1   --lr 0.001  --pretrain torchvision
#CUDA_VISIBLE_DEVICES=0 python train.py metaroom resnet101 all models/metaroom/resnet101/all/noise_0.1 --batch 128 --noise_sd 0.1   --lr 0.001

CUDA_VISIBLE_DEVICES=0 python train.py metaroom resnet101 all models/metaroom/resnet101/all/tv_noise_0.25 --batch 128 --noise_sd 0.25   --lr 0.001  --pretrain torchvision
#CUDA_VISIBLE_DEVICES=0 python train.py metaroom resnet101 all models/metaroom/resnet101/all/noise_0.25 --batch 128 --noise_sd 0.25   --lr 0.001

CUDA_VISIBLE_DEVICES=0 python train.py metaroom resnet101 all models/metaroom/resnet101/all/tv_noise_0.5 --batch 128 --noise_sd 0.5   --lr 0.001  --pretrain torchvision
#CUDA_VISIBLE_DEVICES=0 python train.py metaroom resnet101 all models/metaroom/resnet101/all/noise_0.5 --batch 128 --noise_sd 0.5   --lr 0.001

CUDA_VISIBLE_DEVICES=0 python train.py metaroom resnet101 all models/metaroom/resnet101/all/tv_noise_0.75 --batch 128 --noise_sd 0.75   --lr 0.001  --pretrain torchvision
#CUDA_VISIBLE_DEVICES=0 python train.py metaroom resnet101 all models/metaroom/resnet101/all/noise_0.75 --batch 128 --noise_sd 0.75   --lr 0.001

CUDA_VISIBLE_DEVICES=3 python train.py metaroom resnet50 all models/metaroom/resnet50/all_diffusion/tv_noise_0.5 --batch 16 --noise_sd 0.5   --lr 0.001  --pretrain torchvision  --denoiser diffusion &> train_diffusion_0.5.txt

CUDA_VISIBLE_DEVICES=3 python train.py metaroom resnet50 all models/metaroom/resnet50/all_diffusion/tv_noise_0.25 --batch 16 --noise_sd 0.25   --lr 0.001  --pretrain torchvision  --denoiser diffusion --epochs 1
CUDA_VISIBLE_DEVICES=2 python train.py metaroom resnet50 all models/metaroom/resnet50/all_diffusion/tv_noise_0.75 --batch 8 --noise_sd 0.75   --lr 0.001  --pretrain torchvision  --denoiser diffusion --epochs 2

CUDA_VISIBLE_DEVICES=0 python train.py metaroom resnet101 all models/metaroom/resnet101/all_diffusion/tv_noise_0.5 --batch 16 --noise_sd 0.5   --lr 0.001  --pretrain torchvision  --denoiser diffusion

CUDA_VISIBLE_DEVICES=2 python train.py metaroom resnet50 all models/metaroom/resnet50/all_diffusion/tv_noise_0.25 --batch 16 --noise_sd 0.25   --lr 0.001  --pretrain torchvision  --denoiser diffusion