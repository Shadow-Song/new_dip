import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import StableDiffusionImg2ImgPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import StableDiffusionInpaintPipeline


def load_img2img_pipeline(model_id, device):
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        safety_checker=None,
    )
    return pipe.to(device)


def load_inpaint_pipeline(model_id, device):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        safety_checker=None,
    )
    return pipe.to(device)


def diffusion_img2img(pipe, image, prompt, strength=0.25, guidance_scale=7.5):
    return pipe(
        prompt=prompt,
        image=image,
        strength=strength,
        guidance_scale=guidance_scale,
    ).images[0]


def diffusion_inpaint(pipe, image, mask, prompt, strength=0.35, guidance_scale=7.5):
    return pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        strength=strength,
        guidance_scale=guidance_scale,
    ).images[0]
