import torch
import torch.nn as nn
import torchvision
# import timm 





class Args_imagenet:
    image_size=256
    num_channels=256
    num_res_blocks=2
    num_heads=4
    num_heads_upsample=-1
    num_head_channels=64
    attention_resolutions="32,16,8"
    channel_mult=""
    dropout=0.0
    class_cond=False
    use_checkpoint=False
    use_scale_shift_norm=True
    resblock_updown=True
    use_fp16=False
    use_new_attention_order=False
    clip_denoised=True
    num_samples=10000
    batch_size=16
    use_ddim=False
    model_path=""
    classifier_path=""
    classifier_scale=1.0
    learn_sigma=True
    diffusion_steps=1000
    noise_schedule="linear"
    timestep_respacing=None
    use_kl=False
    predict_xstart=False
    rescale_timesteps=False
    rescale_learned_sigmas=False

class Args_cifar10:
    image_size=32
    num_channels=128
    num_res_blocks=3
    num_heads=4
    num_heads_upsample=-1
    attention_resolutions="16,8"
    dropout=0.3
    learn_sigma=True
    sigma_small=False
    class_cond=False
    diffusion_steps=4000
    noise_schedule="cosine"
    timestep_respacing=""
    use_kl=False
    predict_xstart=False
    rescale_timesteps=True
    rescale_learned_sigmas=True
    use_checkpoint=False
    use_scale_shift_norm=True


class DiffusionRobustModel(nn.Module):
    def __init__(self, classifier, train_flag=False,small=False):
        super().__init__()

        if small:
            from improved_diffusion.script_util import (
                NUM_CLASSES,
                model_and_diffusion_defaults,
                create_model_and_diffusion,
                args_to_dict,
            )
            model, diffusion = create_model_and_diffusion(
                **args_to_dict(Args_cifar10(), model_and_diffusion_defaults().keys())
            )
            model.load_state_dict(
                torch.load("cifar10_diffusion/cifar10_uncond_50M_500K.pt")
            )
        else:
            from guided_diffusion.script_util import (
                NUM_CLASSES,
                model_and_diffusion_defaults,
                create_model_and_diffusion,
                args_to_dict,
            )
            model, diffusion = create_model_and_diffusion(
                **args_to_dict(Args_imagenet(), model_and_diffusion_defaults().keys())
            )
            model.load_state_dict(
                torch.load("imagenet_diffusion/256x256_diffusion_uncond.pt")
            )
        model.eval().cuda()

        self.model = model 
        self.diffusion = diffusion 
        self.train_flag = train_flag
        # # Load the BEiT model
        # classifier = timm.create_model('beit_large_patch16_512', pretrained=True)
        self.classifier = classifier
        if not self.train_flag:
            self.classifier.eval().cuda()
        else:
            self.classifier.train()
        self.small = small



        # self.model = torch.nn.DataParallel(self.model).cuda()
        # self.classifier = torch.nn.DataParallel(self.classifier).cuda()

    def forward(self, x, t):
        # print("%%%%%%%%%%%%%%%%%%%%%%")
        # torchvision.utils.make_grid(x)
        # torchvision.utils.save_image(x, "noised_img.png")
        x_in = x * 2 -1
        imgs = self.denoise(x_in, t)
        # torchvision.utils.make_grid(imgs)
        # torchvision.utils.save_image(imgs, "denoised_img.png")
        # print("$$$$$$$$$$$$$$$$$$", imgs.shape)
        if self.small:
            imgs = torch.nn.functional.interpolate(imgs, (32, 56))
        else:
            imgs = torch.nn.functional.interpolate(imgs, (90, 160))
        # print("!!!!!!!!!!!!!!!!!", imgs.shape)
        # torchvision.utils.make_grid(imgs)
        # torchvision.utils.save_image(imgs, "denoised_img_resize.png")

        imgs = imgs.cuda()
        if not self.train_flag:
            with torch.no_grad():
                out = self.classifier(imgs)
        else:
            out = self.classifier(imgs)

        return out

    def denoise(self, x_start, t, multistep=False):
        t_batch = torch.tensor([t] * len(x_start)).cuda()

        noise = torch.randn_like(x_start)

        x_t_start = self.diffusion.q_sample(x_start=x_start, t=t_batch, noise=noise)

        with torch.no_grad():
            if multistep:
                out = x_t_start
                for i in range(t)[::-1]:
                    print(i)
                    t_batch = torch.tensor([i] * len(x_start)).cuda()
                    out = self.diffusion.p_sample(
                        self.model,
                        out,
                        t_batch,
                        clip_denoised=True
                    )['sample']
            else:
                out = self.diffusion.p_sample(
                    self.model,
                    x_t_start,
                    t_batch,
                    clip_denoised=True
                )['pred_xstart']

        return out