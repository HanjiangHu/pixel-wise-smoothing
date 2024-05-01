# Code for AISTATS 2024: Pixel-wise Smoothing for Certified Robustness against Camera Motion Perturbations

The official code for AISTATS 2024 "Pixel-wise Smoothing for Certified Robustness against Camera Motion Perturbations".


## Preparation
Run the following command to install all packages.

``pip install torchvision seaborn numpy scipy setproctitle matplotlib pandas statsmodels opencv_python torch Pillow python_dateutil setGPU numba open3d cupy-cuda116 tqdm timm transformers``


## Dataset setup and download pretrained model
First follow the README Camera Motion Smoothing (Hu et al. 2022) [here](https://github.com/HanjiangHu/camera-motion-smoothing) to download the dataset and unzip in the root path.
To merge all the training set, run `python generate_all_training_set.py` under the folder `dataset_buildup`. Download class-unconditional diffusion models from [here](https://github.com/openai/guided-diffusion) and put it with the path `imagenet_diffusion/256x256_diffusion_uncond.pt`.


## Model training 
For the model training, run  `bash train.sh` under folder  `./certifiable` for ResNet-50 and ResNet-101 architectures for robust model training with different variances and diffusion model denoisers.


## Certification of PwS, PwS-L, PwS-OF
For the certification, first we run `bash find_required_frames.sh` to find the required number of projected frames for each projection. Then to save computational cost, first run `bash alias.sh` under `./certifiable` to general alias calculation for the margin of projection error and save partitioned images.
Then run `bash diff_certify.sh` under `./certifiable` to general predicted certification files under `./certifiable/data/predict/save_all`.

## Certification of camera motion smoothing (CMS)
For the original camera motion smoothing, run `bash cms_new.sh` under `./certifiable` and the generated predicted files are stored in `./certifiable/predict/cms_new`. To get the certified accuracy results, run the `bash analyze.sh`  under folder `./certifiable`. More can be found through [CMS repo](https://github.com/HanjiangHu/camera-motion-smoothing).

## Benign, empirical robust accuracy  and time efficiency
For the benign and empirical robust accuracy, run the `bash empirical_test.sh`  under folder `./emperical/benign_emperical` and output logs are located in `./emperical/benign_emperical/data`.  Note that `--benign` indicates the benign accuracy while the default is empirical robust accuracy. Change `--pretrain` for correct pretrained models if necessary.
For the average certification time per image, run `python time.py`.

## Citation
If you find the repo useful, please cite:

H. Hu, Z. Liu, L. Li, J. Zhu and D. Zhao
"[Pixel-wise Smoothing for Certified Robustness against Camera Motion Perturbations](https://arxiv.org/abs/2309.13150)", AISTATS 2024
```
@inproceedings{hu2024pixel,
  title={Pixel-wise Smoothing for Certified Robustness against Camera Motion Perturbations},
  author={Hu, Hanjiang and Liu, Zuxin and Li, Linyi and Zhu, Jiacheng and Zhao, Ding},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  pages={217--225},
  year={2024},
  organization={PMLR}
}
```

H. Hu, C. Liu, and D. Zhao "[Robustness Verification for Perception Models against Camera Motion Perturbations](https://files.sri.inf.ethz.ch/wfvml23/papers/paper_17.pdf)", ICML WFVML 2023
```
@inproceedings{hu2023robustness,
  title={Robustness Verification for Perception Models against Camera Motion Perturbations},
  author={Hu, Hanjiang and Liu, Changliu and Zhao, Ding},
  booktitle={ICML Workshop on Formal Verification of Machine Learning (WFVML)},
  year={2023}
}
```

H. Hu, Z. Liu, L. Li, J. Zhu and D. Zhao
"[Robustness Certification of Visual Perception Models via Camera Motion Smoothing](https://arxiv.org/abs/2210.04625)", CoRL 2022

```
@inproceedings{hu2022robustness,
  title={Robustness Certification of Visual Perception Models via Camera Motion Smoothing},
  author={Hu, Hanjiang and Liu, Zuxin and Li, Linyi and Zhu, Jiacheng and Zhao, Ding},
  booktitle={Proceedings of The 6th Conference on Robot Learning},
  year={2022}
}
```

## Reference
> - [camera-motion-smoothing](https://github.com/HanjiangHu/camera-motion-smoothing)
> - [TSS](https://github.com/AI-secure/semantic-randomized-smoothing)
> - [Randomized Smoothing](https://github.com/locuslab/smoothing)
> - [diffusion_denoised_smoothing](https://github.com/ethz-spylab/diffusion_denoised_smoothing)