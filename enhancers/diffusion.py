import json
from pathlib import Path

from PIL import Image


def _dtype_for_device(device):
    import torch

    if device.type == "cuda":
        return torch.float16
    return torch.float32


def load_img2img_pipeline(model_id, device):
    """Load a Stable Diffusion img2img pipeline for prompt-free post-processing."""
    import torch
    from diffusers import StableDiffusionImg2ImgPipeline

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=_dtype_for_device(device),
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)

    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    return pipe


def run_diffusion_img2img(
    input_path,
    output_path,
    model_id="runwayml/stable-diffusion-v1-5",
    device=None,
    strength=0.25,
    guidance_scale=1.0,
    num_inference_steps=25,
    seed=None,
    prompt="",
    negative_prompt="",
):
    """Run img2img diffusion without requiring any user prompt."""
    import torch

    from utils.device_utils import get_device

    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")

    device = get_device() if device is None else torch.device(device)
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.open(input_path).convert("RGB")
    pipe = load_img2img_pipeline(model_id, device)

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    ).images[0]

    result.save(output_path)
    metadata = {
        "model_id": model_id,
        "device": str(device),
        "input": str(input_path),
        "output": str(output_path),
        "strength": strength,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "seed": seed,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
    }
    (output_path.parent / "diffusion_config.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    )
    return metadata
