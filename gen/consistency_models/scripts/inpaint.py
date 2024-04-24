"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from cm import dist_util, logger
from cm.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from cm.random_util import get_generator
from cm.karras_diffusion import iterative_inpainting

# for image loading
import torchvision.transforms as transforms
from PIL import Image

from torchvision.utils import save_image

@th.no_grad()
def custom_iterative_inpainting(
    distiller,
    images,
    masks,
    x,
    ts,
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
    generator=None,
):
    """
    Perform inpainting on given images using a mask and a diffusion model.

    Args:
        distiller: The diffusion model used for generating the inpaintings.
        images: A batch of images (tensor) to inpaint.
        masks: A batch of masks (tensor) where 1 indicates the area to keep and 0 the area to inpaint.
        x: The initial noise tensor.
        ts: A list of timesteps for the diffusion process.
        t_min: Minimum noise level.
        t_max: Maximum noise level.
        rho: A parameter for the noise schedule.
        steps: Number of diffusion steps.
        generator: Random number generator for noise.

    Returns:
        Tuple of tensors (final image after inpainting, initial masked images)
    """
    # Ensure mask is in the correct format
    masks = masks.float()
    
    def replacement(x0, x1):
        x_mix = x0 * masks + x1 * (1 - masks)
        return x_mix

    images = replacement(images, -th.ones_like(images))

    # Convert the time schedule based on the given rho
    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])

    # rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)

    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        x0 = distiller(x, t * s_in)
        x0 = th.clamp(x0, -1.0, 1.0)
        x0 = replacement(images, x0)
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)
        x = x0 + generator.randn_like(x) * np.sqrt(next_t**2 - t_min**2)

    return x, images

def load_incomplete_image(size):
    """
    Simple function to load only one image and one mask, prepares it for 
    """
    image_path = '../../000000008532.jpg'
    image = Image.open(image_path)
    image = image.convert("RGB")  # Ensure it's RGB

    # Load or create your mask
    mask_path = '../../mask_000000008532.jpg'
    mask = Image.open(mask_path).convert("L")

    # Convert images and masks to PyTorch tensors
    transform = transforms.ToTensor()
    cropper = transforms.Resize((size, size))
    image = cropper(image)
    image = np.array(image)
    image = image.astype(np.float32) / 127.5 - 1
    image -= [0.0937, 0.0885, 0.0882]
    
    mask = cropper(mask)
    image_tensor = transform(image)

    mask_tensor = transform(mask).float()  # Ensure mask is float
    mask_tensor = mask_tensor.repeat(3, 1, 1)
    mask_tensor = (mask_tensor > 0).float()  # Ensure mask is binary (0 or 1)
    # mask_tensor = 1 - mask_tensor

    # image_tensor = image_tensor.to(th.float32) / 127.5 - 1
    # image_tensor -= th.tensor([0.0937, 0.0885, 0.0882])

    return image_tensor.cuda().unsqueeze(0), mask_tensor.cuda().unsqueeze(0)
    # Apply the mask to the image
    # This zeros out the regions of the image where the mask is zero
    incomplete_image = image_tensor * mask_tensor
    return incomplete_image

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    if "consistency" in args.training_mode:
        distillation = True
    else:
        distillation = False

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
        distillation=distillation,
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    if args.sampler == "multistep":
        assert len(args.ts) > 0
        ts = tuple(int(x) for x in args.ts.split(","))
    else:
        ts = None

    all_images = []
    all_labels = []
    generator = get_generator(args.generator, args.num_samples, args.seed)

    # start by only filling in one image the whole time
    images, masks = load_incomplete_image(args.image_size)
    print(images.shape)
    
    shape = (args.batch_size, 3, args.image_size, args.image_size)
    device='cuda'
    x_T = generator.randn(*shape, device=device) * args.sigma_max

    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes

        sample, images = custom_iterative_inpainting(
            model,
            images= images,
            masks = masks,
            x=x_T,  # Starting point
            ts=ts,
            t_min=args.sigma_min,
            t_max=args.sigma_max,
            rho=7.0,  # Customize as necessary
            steps=args.steps,
            generator=generator
        )
        # print(sample.shape)
        save_image(sample, "trsitan_test.png")
        
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        
        sample = sample.contiguous()
        # save_image(sample, "trsitan_test.png")

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        # out_path = '~/scratch/seg-replace'
        out_path = f"samples_{shape_str}.npz"
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        training_mode="edm",
        generator="determ",
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        sampler="heun",
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        steps=40,
        model_path="",
        seed=42,
        ts="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
