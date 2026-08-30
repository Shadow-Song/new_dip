import argparse
from pathlib import Path
from PIL import Image

from utils.device_utils import get_device
from functions.inpainting import run_inpainting
from diffusion import load_inpaint_pipeline, diffusion_inpaint
from postprocess import paste_known_region


def run_dip_diffusion_inpainting(
    image_path,
    mask_path,
    output_dir,
    prompt,
    model_id="runwayml/stable-diffusion-inpainting",
    dip_iters=3000,
    strength=0.35,
    guidance_scale=7.5,
    device=None,
):
    device = get_device() if device is None else device
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dip_dir = output_dir / "dip"
    diff_dir = output_dir / "diffusion"
    diff_dir.mkdir(parents=True, exist_ok=True)

    dip_summary = run_inpainting(
        input_path=image_path,
        mask_path=mask_path,
        output_dir=dip_dir,
        num_iter=dip_iters,
        device=str(device),
        verbose=True,
    )

    dip_image_path = dip_summary["images"]["inpainted"]
    dip_image = Image.open(dip_image_path).convert("RGB")
    original = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    pipe = load_inpaint_pipeline(model_id, device)

    generated = diffusion_inpaint(
        pipe=pipe,
        image=dip_image,
        mask=mask,
        prompt=prompt,
        strength=strength,
        guidance_scale=guidance_scale,
    )

    final = paste_known_region(original, generated, mask)

    generated.save(diff_dir / "diffusion.png")
    final.save(output_dir / "final.png")

    return {
        "dip": dip_summary,
        "diffusion": str(diff_dir / "diffusion.png"),
        "final": str(output_dir / "final.png"),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="inpainting")
    parser.add_argument("--input", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--output-dir", default="result/hybrid_inpainting")
    parser.add_argument("--prompt", default="a realistic photograph")
    parser.add_argument("--model-id", default="runwayml/stable-diffusion-inpainting")
    parser.add_argument("--dip-iters", type=int, default=3000)
    parser.add_argument("--strength", type=float, default=0.35)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.task != "inpainting":
        raise ValueError("First hybrid version only supports inpainting")

    result = run_dip_diffusion_inpainting(
        image_path=args.input,
        mask_path=args.mask,
        output_dir=args.output_dir,
        prompt=args.prompt,
        model_id=args.model_id,
        dip_iters=args.dip_iters,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
    )

    print(result)


if __name__ == "__main__":
    main()
