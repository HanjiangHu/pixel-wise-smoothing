CUDA_VISIBLE_DEVICES=0 python certify.py metaroom ~/projective_transformation/certifiable/models/all/resnet50/all/tv_noise_0.5/checkpoint.pth.tar 0.01 resolvable_tz data/predict/cms_new/metaroom/resnet50/tz/tv_noise_0.1_0.01_0.5 --batch 200 --N0 100 --N 1000
CUDA_VISIBLE_DEVICES=0 python certify.py metaroom ~/projective_transformation/certifiable/models/all/resnet101/all/tv_noise_0.5/checkpoint.pth.tar 0.01 resolvable_tz data/predict/cms_new/metaroom/resnet101/tz/tv_noise_0.1_0.01_0.5 --batch 200 --N0 100 --N 1000
CUDA_VISIBLE_DEVICES=0 python certify.py metaroom ~/projective_transformation/certifiable/models/all/resnet50/all/tv_noise_0.5/checkpoint.pth.tar 0.1 resolvable_tz data/predict/cms_new/metaroom/resnet50/tz/tv_noise_0.01_0.1_0.5 --batch 200 --N0 100 --N 1000 # smoothing 0.1, attack 0.01

CUDA_VISIBLE_DEVICES=0 python certify.py metaroom ~/projective_transformation/certifiable/models/all/resnet50/all/tv_noise_0.5/checkpoint.pth.tar 0.04363323 resolvable_ry data/predict/cms_new/metaroom/resnet50/ry/tv_noise_0.25_2.5_0.5 --batch 200 --N0 100 --N 1000

CUDA_VISIBLE_DEVICES=0 python certify.py metaroom ~/projective_transformation/certifiable/models/all/resnet50/all/tv_noise_0.5/checkpoint.pth.tar 0.05 resolvable_ty data/predict/cms_new/metaroom/resnet50/ty/tv_noise_0.005_0.05_0.5 --batch 200 --N0 100 --N 1000

CUDA_VISIBLE_DEVICES=0 python certify.py metaroom ~/projective_transformation/certifiable/models/all/resnet50/all/tv_noise_0.5/checkpoint.pth.tar 0.05 resolvable_tx data/predict/cms_new/metaroom/resnet50/tx/tv_noise_0.005_0.05_0.5 --batch 200 --N0 100 --N 1000

CUDA_VISIBLE_DEVICES=0 python certify.py metaroom ~/projective_transformation/certifiable/models/all/resnet50/all/tv_noise_0.5/checkpoint.pth.tar 0.04363323 resolvable_rx data/predict/cms_new/metaroom/resnet50/rx/tv_noise_0.25_2.5_0.5 --batch 200 --N0 100 --N 1000
 
CUDA_VISIBLE_DEVICES=0 python certify.py metaroom ~/projective_transformation/certifiable/models/all/resnet50/all/tv_noise_0.5/checkpoint.pth.tar 0.122173 resolvable_rz data/predict/cms_new/metaroom/resnet50/rz/tv_noise_0.7_7_0.5 --batch 200 --N0 100 --N 1000

CUDA_VISIBLE_DEVICES=0 python certify.py metaroom ~/projective_transformation/certifiable/models/all/resnet50/all/tv_noise_0.5/checkpoint.pth.tar 0.04363323 resolvable_rz data/predict/cms_new/metaroom/resnet50/rz/tv_noise_0.25_2.5_0.5 --batch 200 --N0 100 --N 1000

CUDA_VISIBLE_DEVICES=0 python certify.py metaroom ~/projective_transformation/certifiable/models/all/resnet101/all/tv_noise_0.5/checkpoint.pth.tar 0.1 resolvable_tz data/predict/cms_new/metaroom/resnet101/tz/tv_noise_0.01_0.1_0.5 --batch 200 --N0 100 --N 1000

CUDA_VISIBLE_DEVICES=0 python certify.py metaroom ~/projective_transformation/certifiable/models/all/resnet101/all/tv_noise_0.5/checkpoint.pth.tar 0.04363323 resolvable_ry data/predict/cms_new/metaroom/resnet101/ry/tv_noise_0.25_2.5_0.5 --batch 200 --N0 100 --N 1000



CUDA_VISIBLE_DEVICES=1 python certify.py metaroom ~/projective_transformation/certifiable/models/all/resnet50/all/tv_noise_0.5/checkpoint.pth.tar 0.1 resolvable_tz data/predict/cms_new/metaroom/resnet50/tz/tv_noise_0.1_0.5 --batch 200 --N0 100 --N 1000
CUDA_VISIBLE_DEVICES=0 python certify.py metaroom ~/projective_transformation/certifiable/models/all/resnet50/all/tv_noise_0.5/checkpoint.pth.tar 0.1 resolvable_tz data/predict/cms_new/metaroom/resnet50/tz/tv_noise_0.1_0.5 --batch 200 --N0 100 --N 1000

CUDA_VISIBLE_DEVICES=1 python certify.py metaroom ~/projective_transformation/certifiable/models/metaroom/resnet50/all/tv_noise_0.5/checkpoint.pth.tar 0.04363323 resolvable_ry data/predict/cms_new/metaroom/resnet50/ry/tv_noise_2.5 --batch 200 --N0 100 --N 1000

CUDA_VISIBLE_DEVICES=1 python certify.py metaroom ~/projective_transformation/certifiable/models/metaroom/resnet50/all/tv_noise_0.5/checkpoint.pth.tar 0.05 resolvable_ty data/predict/cms_new/metaroom/resnet50/ty/tv_noise_0.05 --batch 200 --N0 100 --N 1000

CUDA_VISIBLE_DEVICES=1 python certify.py metaroom ~/projective_transformation/certifiable/models/metaroom/resnet50/all/tv_noise_0.5/checkpoint.pth.tar 0.05 resolvable_tx data/predict/cms_new/metaroom/resnet50/tx/tv_noise_0.05 --batch 200 --N0 100 --N 1000

CUDA_VISIBLE_DEVICES=1 python certify.py metaroom ~/projective_transformation/certifiable/models/metaroom/resnet50/all/tv_noise_0.5/checkpoint.pth.tar 0.04363323 resolvable_rx data/predict/cms_new/metaroom/resnet50/rx/tv_noise_2.5 --batch 200 --N0 100 --N 1000

CUDA_VISIBLE_DEVICES=1 python certify.py metaroom ~/projective_transformation/certifiable/models/metaroom/resnet50/all/tv_noise_0.5/checkpoint.pth.tar 0.122173 resolvable_rz data/predict/cms_new/metaroom/resnet50/rz/tv_noise_7 --batch 200 --N0 100 --N 1000
CUDA_VISIBLE_DEVICES=0 python certify.py metaroom ~/projective_transformation/certifiable/models/all/resnet50/all/tv_noise_0.5/checkpoint.pth.tar 0.1 resolvable_tz data/predict/cms_new/metaroom/resnet50/tz/tv_noise_0.1_0.5 --batch 200 --N0 100 --N 1000

CUDA_VISIBLE_DEVICES=0 python certify.py metaroom ~/projective_transformation/certifiable/models/all/resnet50/all/tv_noise_0.5/checkpoint.pth.tar 0.04363323 resolvable_ry data/predict/cms_new/metaroom/resnet50/ry/tv_noise_2.5_0.5 --batch 200 --N0 100 --N 1000

CUDA_VISIBLE_DEVICES=0 python certify.py metaroom ~/projective_transformation/certifiable/models/all/resnet50/all/tv_noise_0.5/checkpoint.pth.tar 0.05 resolvable_ty data/predict/cms_new/metaroom/resnet50/ty/tv_noise_0.05_0.5 --batch 200 --N0 100 --N 1000

CUDA_VISIBLE_DEVICES=0 python certify.py metaroom ~/projective_transformation/certifiable/models/all/resnet50/all/tv_noise_0.5/checkpoint.pth.tar 0.05 resolvable_tx data/predict/cms_new/metaroom/resnet50/tx/tv_noise_0.05_0.5 --batch 200 --N0 100 --N 1000

CUDA_VISIBLE_DEVICES=0 python certify.py metaroom ~/projective_transformation/certifiable/models/all/resnet50/all/tv_noise_0.5/checkpoint.pth.tar 0.04363323 resolvable_rx data/predict/cms_new/metaroom/resnet50/rx/tv_noise_2.5_0.5 --batch 200 --N0 100 --N 1000

CUDA_VISIBLE_DEVICES=0 python certify.py metaroom ~/projective_transformation/certifiable/models/all/resnet50/all/tv_noise_0.5/checkpoint.pth.tar 0.122173 resolvable_rz data/predict/cms_new/metaroom/resnet50/rz/tv_noise_7_0.5_38 --batch 200 --N0 100 --N 1000 --start 38

CUDA_VISIBLE_DEVICES=0 python certify.py metaroom ~/projective_transformation/certifiable/models/all/resnet50/all/tv_noise_0.5/checkpoint.pth.tar 0.04363323 resolvable_rz data/predict/cms_new/metaroom/resnet50/rz/tv_noise_2.5_0.5 --batch 200 --N0 100 --N 1000



#CUDA_VISIBLE_DEVICES=0 python certify.py metaroom ~/projective_transformation/certifiable/models/all/resnet101/all/tv_noise_0.5/checkpoint.pth.tar 0.1 resolvable_tz data/predict/cms_new/metaroom/resnet101/tz/tv_noise_0.1_0.5 --batch 200 --N0 100 --N 1000

CUDA_VISIBLE_DEVICES=0 python certify.py metaroom ~/projective_transformation/certifiable/models/all/resnet101/all/tv_noise_0.5/checkpoint.pth.tar 0.04363323 resolvable_ry data/predict/cms_new/metaroom/resnet101/ry/tv_noise_2.5_0.5 --batch 200 --N0 100 --N 1000

CUDA_VISIBLE_DEVICES=0 python certify.py metaroom ~/projective_transformation/certifiable/models/all/resnet101/all/tv_noise_0.5/checkpoint.pth.tar 0.05 resolvable_ty data/predict/cms_new/metaroom/resnet101/ty/tv_noise_0.05_0.5 --batch 200 --N0 100 --N 1000
CUDA_VISIBLE_DEVICES=0 python certify.py metaroom ~/projective_transformation/certifiable/models/all/resnet50/all/tv_noise_0.75/checkpoint.pth.tar 0.01 resolvable_tz data/predict/cms_new/metaroom/resnet50/tz/tv_noise_0.01_0.75_7k_23 --batch 1000 --N0 500 --N 7000 --start 23

CUDA_VISIBLE_DEVICES=0 python certify.py metaroom ~/projective_transformation/certifiable/models/all/resnet50/all/tv_noise_0.75/checkpoint.pth.tar 0.01 resolvable_tz data/predict/cms_new/metaroom/resnet50/tz/tv_noise_0.1_0.01_0.75 --batch 200 --N0 100 --N 1000
CUDA_VISIBLE_DEVICES=0 python certify.py metaroom ~/projective_transformation/certifiable/models/all/resnet101/all/tv_noise_0.75/checkpoint.pth.tar 0.01 resolvable_tz data/predict/cms_new/metaroom/resnet101/tz/tv_noise_0.1_0.01_0.75 --batch 200 --N0 100 --N 1000
CUDA_VISIBLE_DEVICES=1 python certify.py metaroom ~/projective_transformation/certifiable/models/all/resnet50/all/tv_noise_0.75/checkpoint.pth.tar 0.1 resolvable_tz data/predict/cms_new/metaroom/resnet50/tz/tv_noise_0.01_0.1_0.75 --batch 200 --N0 100 --N 1000

CUDA_VISIBLE_DEVICES=1 python certify.py metaroom ~/projective_transformation/certifiable/models/all/resnet50/all/tv_noise_0.75/checkpoint.pth.tar 0.04363323 resolvable_ry data/predict/cms_new/metaroom/resnet50/ry/tv_noise_0.25_2.5_0.75 --batch 200 --N0 100 --N 1000

CUDA_VISIBLE_DEVICES=1 python certify.py metaroom ~/projective_transformation/certifiable/models/all/resnet50/all/tv_noise_0.75/checkpoint.pth.tar 0.05 resolvable_ty data/predict/cms_new/metaroom/resnet50/ty/tv_noise_0.005_0.05_0.75 --batch 200 --N0 100 --N 1000

CUDA_VISIBLE_DEVICES=1 python certify.py metaroom ~/projective_transformation/certifiable/models/all/resnet50/all/tv_noise_0.75/checkpoint.pth.tar 0.05 resolvable_tx data/predict/cms_new/metaroom/resnet50/tx/tv_noise_0.005_0.05_0.75 --batch 200 --N0 100 --N 1000

CUDA_VISIBLE_DEVICES=1 python certify.py metaroom ~/projective_transformation/certifiable/models/all/resnet50/all/tv_noise_0.75/checkpoint.pth.tar 0.04363323 resolvable_rx data/predict/cms_new/metaroom/resnet50/rx/tv_noise_0.25_2.5_0.75 --batch 200 --N0 100 --N 1000

CUDA_VISIBLE_DEVICES=1 python certify.py metaroom ~/projective_transformation/certifiable/models/all/resnet50/all/tv_noise_0.75/checkpoint.pth.tar 0.122173 resolvable_rz data/predict/cms_new/metaroom/resnet50/rz/tv_noise_0.7_7_0.75 --batch 200 --N0 100 --N 1000

CUDA_VISIBLE_DEVICES=1 python certify.py metaroom ~/projective_transformation/certifiable/models/all/resnet50/all/tv_noise_0.75/checkpoint.pth.tar 0.04363323 resolvable_rz data/predict/cms_new/metaroom/resnet50/rz/tv_noise_0.25_2.5_0.75 --batch 200 --N0 100 --N 1000

CUDA_VISIBLE_DEVICES=1 python certify.py metaroom ~/projective_transformation/certifiable/models/all/resnet101/all/tv_noise_0.75/checkpoint.pth.tar 0.1 resolvable_tz data/predict/cms_new/metaroom/resnet101/tz/tv_noise_0.01_0.1_0.75 --batch 200 --N0 100 --N 1000

CUDA_VISIBLE_DEVICES=1 python certify.py metaroom ~/projective_transformation/certifiable/models/all/resnet101/all/tv_noise_0.75/checkpoint.pth.tar 0.04363323 resolvable_ry data/predict/cms_new/metaroom/resnet101/ry/tv_noise_0.25_2.5_0.75 --batch 200 --N0 100 --N 1000

CUDA_VISIBLE_DEVICES=1 python certify.py metaroom ~/projective_transformation/certifiable/models/all/resnet50/all/tv_noise_0.75/checkpoint.pth.tar 0.04363323 resolvable_rz data/predict/cms_new/metaroom/resnet50/rz/tv_noise_2.5_0.75 --batch 200 --N0 100 --N 1000
#CUDA_VISIBLE_DEVICES=1 python certify.py metaroom ~/projective_transformation/certifiable/models/all/resnet101/all/tv_noise_0.75/checkpoint.pth.tar 0.1 resolvable_tz data/predict/cms_new/metaroom/resnet101/tz/tv_noise_0.1_0.75 --batch 200 --N0 100 --N 1000

CUDA_VISIBLE_DEVICES=1 python certify.py metaroom ~/projective_transformation/certifiable/models/all/resnet101/all/tv_noise_0.75/checkpoint.pth.tar 0.04363323 resolvable_ry data/predict/cms_new/metaroom/resnet101/ry/tv_noise_2.5_0.75 --batch 200 --N0 100 --N 1000

CUDA_VISIBLE_DEVICES=1 python certify.py metaroom ~/projective_transformation/certifiable/models/all/resnet101/all/tv_noise_0.75/checkpoint.pth.tar 0.05 resolvable_ty data/predict/cms_new/metaroom/resnet101/ty/tv_noise_0.05_0.75 --batch 200 --N0 100 --N 1000
CUDA_VISIBLE_DEVICES=1 python certify.py metaroom ~/projective_transformation/certifiable/models/all/resnet50/all/tv_noise_0.75/checkpoint.pth.tar 0.01 resolvable_tz data/predict/cms_new/metaroom/resnet50/tz/tv_noise_0.01_0.75_24 --batch 200 --N0 100 --N 1000 --max 24

CUDA_VISIBLE_DEVICES=1 python certify.py metaroom ~/projective_transformation/certifiable/models/all/resnet101/all/tv_noise_0.75/checkpoint.pth.tar 0.05 resolvable_tx data/predict/cms_new/metaroom/resnet101/tx/tv_noise_0.05_0.75_32 --batch 200 --N0 100 --N 1000 --start 32

CUDA_VISIBLE_DEVICES=1 python certify.py metaroom ~/projective_transformation/certifiable/models/all/resnet101/all/tv_noise_0.75/checkpoint.pth.tar 0.04363323 resolvable_rx data/predict/cms_new/metaroom/resnet101/rx/tv_noise_2.5_0.75_45 --batch 200 --N0 100 --N 1000 --start 45

CUDA_VISIBLE_DEVICES=1 python certify.py metaroom ~/projective_transformation/certifiable/models/all/resnet101/all/tv_noise_0.75/checkpoint.pth.tar 0.04363323 resolvable_rz data/predict/cms_new/metaroom/resnet101/rz/tv_noise_2.5_0.75 --batch 200 --N0 100 --N 1000

CUDA_VISIBLE_DEVICES=0 python certify.py metaroom ~/projective_transformation/certifiable/models/all/resnet101/all/tv_noise_0.5/checkpoint.pth.tar 0.05 resolvable_tx data/predict/cms_new/metaroom/resnet101/tx/tv_noise_0.05_0.5_36 --batch 200 --N0 100 --N 1000 --start 36

CUDA_VISIBLE_DEVICES=0 python certify.py metaroom ~/projective_transformation/certifiable/models/all/resnet101/all/tv_noise_0.5/checkpoint.pth.tar 0.04363323 resolvable_rx data/predict/cms_new/metaroom/resnet101/rx/tv_noise_2.5_0.5 --batch 200 --N0 100 --N 1000
#
#CUDA_VISIBLE_DEVICES=0 python certify.py metaroom ~/projective_transformatio
