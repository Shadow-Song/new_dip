from __future__ import print_function

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _display_path(path):
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _default_output_dir(output_dir):
    if output_dir is not None:
        return _resolve_path(output_dir)
    return REPO_ROOT / "result" / datetime.now().strftime("%Y%m%d_%H%M%S") / "dip_diffusion"


def run_dip_diffusion(
    input_path="data/sr/zebra_GT.png",
    factor=4,
    output_dir=None,
    dip_num_iter=None,
    dip_lr=0.01,
    dip_device=None,
    dip_save_every=0,
    model_id="runwayml/stable-diffusion-v1-5",
    strength=0.25,
    guidance_scale=1.0,
    num_inference_steps=25,
    blend_alpha=0.35,
    seed=None,
    verbose=True,
):
    """Run DIP super-resolution followed by prompt-free diffusion post-processing."""
    from enhancers.diffusion import run_diffusion_img2img
    from enhancers.image_ops import blend_images, make_comparison, resize_to_match
    from functions.super_resolution import run_super_resolution

    input_path = _resolve_path(input_path)
    output_dir = _default_output_dir(output_dir)
    dip_dir = output_dir / "dip"
    diffusion_dir = output_dir / "diffusion"
    images_dir = output_dir / "images"

    output_dir.mkdir(parents=True, exist_ok=True)
    diffusion_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "input_path": _display_path(input_path),
        "factor": factor,
        "dip_num_iter": dip_num_iter,
        "dip_lr": dip_lr,
        "dip_device": dip_device,
        "dip_save_every": dip_save_every,
        "model_id": model_id,
        "strength": strength,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "blend_alpha": blend_alpha,
        "seed": seed,
        "output_dir": _display_path(output_dir),
        "prompt": "",
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")

    dip_summary = run_super_resolution(
        input_path=str(input_path),
        factor=factor,
        output_dir=dip_dir,
        num_iter=dip_num_iter,
        lr=dip_lr,
        device=dip_device,
        save_every=dip_save_every,
        seed=seed,
        verbose=verbose,
    )

    dip_image = _resolve_path(dip_summary["images"]["deep_prior_hr"])
    diffusion_raw = diffusion_dir / "diffusion_raw.png"
    diffusion_resized = diffusion_dir / "diffusion_resized.png"
    final_blend = images_dir / "final_blend.png"
    comparison = images_dir / "comparison.png"

    diffusion_summary = run_diffusion_img2img(
        input_path=dip_image,
        output_path=diffusion_raw,
        model_id=model_id,
        device=dip_device,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
        prompt="",
        negative_prompt="",
    )

    resize_to_match(diffusion_raw, dip_image, diffusion_resized)
    blend_images(dip_image, diffusion_resized, final_blend, alpha=blend_alpha)

    comparison_inputs = [
        ("DIP", dip_image),
        ("Diffusion", diffusion_resized),
        ("Blend", final_blend),
    ]
    bicubic_path = _resolve_path(dip_summary["images"].get("bicubic", ""))
    if bicubic_path.exists():
        comparison_inputs.insert(0, ("Bicubic", bicubic_path))
    make_comparison(comparison, comparison_inputs)

    summary = {
        "status": "success",
        "output_dir": _display_path(output_dir),
        "dip": dip_summary,
        "diffusion": diffusion_summary,
        "images": {
            "dip": _display_path(dip_image),
            "diffusion_raw": _display_path(diffusion_raw),
            "diffusion_resized": _display_path(diffusion_resized),
            "final_blend": _display_path(final_blend),
            "comparison": _display_path(comparison),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run DIP followed by prompt-free diffusion post-processing.")
    parser.add_argument("--input", default="data/sr/zebra_GT.png")
    parser.add_argument("--factor", type=int, default=4)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-iter", type=int, default=None, help="DIP optimization iterations.")
    parser.add_argument("--lr", type=float, default=0.01, help="DIP learning rate.")
    parser.add_argument("--device", default=None, help="cuda, mps, cpu, or auto when omitted.")
    parser.add_argument("--dip-save-every", type=int, default=0)
    parser.add_argument("--model-id", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--strength", type=float, default=0.25)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--num-inference-steps", type=int, default=25)
    parser.add_argument("--blend-alpha", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = run_dip_diffusion(
        input_path=args.input,
        factor=args.factor,
        output_dir=args.output_dir,
        dip_num_iter=args.num_iter,
        dip_lr=args.lr,
        dip_device=args.device,
        dip_save_every=args.dip_save_every,
        model_id=args.model_id,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        blend_alpha=args.blend_alpha,
        seed=args.seed,
        verbose=not args.quiet,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
