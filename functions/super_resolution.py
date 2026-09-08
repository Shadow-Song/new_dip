from __future__ import print_function

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _display_path(path):
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _default_output_dir(output_dir=None):
    if output_dir is not None:
        return _resolve_path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "result" / timestamp / "super_resolution"


def _psnr(x, y):
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr

    return float(compare_psnr(x, y, data_range=1.0))


def save_image_np(path, image_np):
    import numpy as np

    from utils.common_utils import np_to_pil

    path.parent.mkdir(parents=True, exist_ok=True)
    np_to_pil(np.clip(image_np, 0, 1)).save(path)


def run_super_resolution(
    input_path="data/sr/zebra_GT.png",
    factor=4,
    output_dir=None,
    imsize=-1,
    enforse_div32="CROP",
    input_depth=32,
    input_mode="noise",
    pad="reflection",
    opt_over="net",
    kernel_type="lanczos2",
    lr=0.01,
    tv_weight=0.0,
    optimizer="adam",
    num_iter=None,
    reg_noise_std=None,
    skip_n33d=128,
    skip_n33u=128,
    skip_n11=4,
    num_scales=5,
    upsample_mode="bilinear",
    device=None,
    save_every=100,
    seed=None,
    verbose=True,
):
    """Run the super-resolution experiment from super-resolution.ipynb.

    Args:
        input_path: High-resolution source image. The function creates the
            low-resolution observation internally, matching the original notebook.
        factor: Super-resolution factor. The original notebook uses 4 or 8.
        output_dir: Directory where images, metrics, and config are saved.
        imsize: Optional input resize. Use -1 for original size.
        enforse_div32: Keep the notebook's original parameter spelling.
        num_iter: Optimization iterations. Defaults to notebook values per factor.
        reg_noise_std: Input-noise regularization. Defaults to notebook values per factor.
        device: "cuda", "mps", "cpu", or None for automatic selection.
        save_every: Save an intermediate HR image every N iterations. Use 0 to disable.
        seed: Optional random seed for reproducibility.

    Returns:
        A dictionary with output paths, final metrics, and selected parameters.
    """
    import numpy as np
    import torch

    from models import get_net
    from models.downsampler import Downsampler
    from utils.common_utils import get_noise, get_params, torch_to_np, optimize
    from utils.device_utils import get_device
    from utils.sr_utils import get_baselines, load_LR_HR_imgs_sr, put_in_center, tv_loss

    if factor < 2:
        raise ValueError("factor must be an integer greater than or equal to 2")

    if factor == 2:
        default_num_iter = 2000
        default_reg_noise_std = 0.03
    elif factor == 4:
        default_num_iter = 2000
        default_reg_noise_std = 0.03
    elif factor == 8:
        default_num_iter = 4000
        default_reg_noise_std = 0.05
    else:
        default_num_iter = 2000
        default_reg_noise_std = 0.03

    if num_iter is None:
        num_iter = default_num_iter
    if reg_noise_std is None:
        reg_noise_std = default_reg_noise_std

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    device = get_device() if device is None else torch.device(device)
    input_path = _resolve_path(input_path)
    output_dir = _default_output_dir(output_dir)
    images_dir = output_dir / "images"
    progress_dir = output_dir / "progress"
    metrics_path = output_dir / "metrics.csv"
    config_path = output_dir / "config.json"
    summary_path = output_dir / "summary.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    if save_every:
        progress_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "input_path": _display_path(input_path),
        "factor": factor,
        "imsize": imsize,
        "enforse_div32": enforse_div32,
        "input_depth": input_depth,
        "input_mode": input_mode,
        "pad": pad,
        "opt_over": opt_over,
        "kernel_type": kernel_type,
        "lr": lr,
        "tv_weight": tv_weight,
        "optimizer": optimizer,
        "num_iter": num_iter,
        "reg_noise_std": reg_noise_std,
        "skip_n33d": skip_n33d,
        "skip_n33u": skip_n33u,
        "skip_n11": skip_n11,
        "num_scales": num_scales,
        "upsample_mode": upsample_mode,
        "device": str(device),
        "save_every": save_every,
        "seed": seed,
        "output_dir": _display_path(output_dir),
    }
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")

    imgs = load_LR_HR_imgs_sr(str(input_path), imsize, factor, enforse_div32)
    imgs["bicubic_np"], imgs["sharp_np"], imgs["nearest_np"] = get_baselines(
        imgs["LR_pil"],
        imgs["HR_pil"],
    )

    save_image_np(images_dir / "input_hr.png", imgs["HR_np"])
    save_image_np(images_dir / "input_lr.png", imgs["LR_np"])
    save_image_np(images_dir / "bicubic.png", imgs["bicubic_np"])
    save_image_np(images_dir / "nearest.png", imgs["nearest_np"])
    save_image_np(images_dir / "sharp.png", imgs["sharp_np"])

    net_input = get_noise(
        input_depth,
        input_mode,
        (imgs["HR_pil"].size[1], imgs["HR_pil"].size[0]),
    ).to(device).detach()

    net = get_net(
        input_depth,
        "skip",
        pad,
        skip_n33d=skip_n33d,
        skip_n33u=skip_n33u,
        skip_n11=skip_n11,
        num_scales=num_scales,
        upsample_mode=upsample_mode,
    ).to(device)

    mse = torch.nn.MSELoss().to(device)
    img_LR_var = torch.from_numpy(imgs["LR_np"])[None, :].to(device)
    downsampler = Downsampler(
        n_planes=3,
        factor=factor,
        kernel_type=kernel_type,
        phase=0.5,
        preserve_size=True,
    ).to(device)

    psnr_history = []
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    iteration = {"value": 0}

    with metrics_path.open("w", newline="") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=["iteration", "psnr_lr", "psnr_hr", "loss"])
        writer.writeheader()

        def closure():
            if reg_noise_std > 0:
                current_input = net_input_saved + (noise.normal_() * reg_noise_std)
            else:
                current_input = net_input_saved

            out_HR = net(current_input)
            out_LR = downsampler(out_HR)

            total_loss = mse(out_LR, img_LR_var)
            if tv_weight > 0:
                total_loss = total_loss + tv_weight * tv_loss(out_HR)

            total_loss.backward()

            out_LR_np = torch_to_np(out_LR)
            out_HR_np = torch_to_np(out_HR)
            psnr_LR = _psnr(imgs["LR_np"], out_LR_np)
            psnr_HR = _psnr(imgs["HR_np"], out_HR_np)
            loss_value = float(total_loss.detach().cpu().item())
            current_iter = iteration["value"]

            writer.writerow(
                {
                    "iteration": current_iter,
                    "psnr_lr": psnr_LR,
                    "psnr_hr": psnr_HR,
                    "loss": loss_value,
                }
            )
            metrics_file.flush()
            psnr_history.append([psnr_LR, psnr_HR])

            if verbose:
                print(
                    "Iteration %05d    PSNR_LR %.3f   PSNR_HR %.3f"
                    % (current_iter, psnr_LR, psnr_HR),
                    "\r",
                    end="",
                )

            if save_every and current_iter % save_every == 0:
                save_image_np(progress_dir / ("iter_%05d.png" % current_iter), out_HR_np)

            iteration["value"] += 1
            return total_loss

        params = get_params(opt_over, net, net_input)
        optimize(optimizer, params, closure, lr, num_iter)

    if verbose:
        print()

    with torch.no_grad():
        out_HR_np = np.clip(torch_to_np(net(net_input_saved)), 0, 1)

    result_deep_prior = put_in_center(out_HR_np, imgs["orig_np"].shape[1:])
    save_image_np(images_dir / "deep_prior_hr.png", out_HR_np)
    save_image_np(images_dir / "deep_prior_centered.png", result_deep_prior)

    summary = {
        "status": "success",
        "output_dir": _display_path(output_dir),
        "images": {
            "input_hr": _display_path(images_dir / "input_hr.png"),
            "input_lr": _display_path(images_dir / "input_lr.png"),
            "bicubic": _display_path(images_dir / "bicubic.png"),
            "nearest": _display_path(images_dir / "nearest.png"),
            "sharp": _display_path(images_dir / "sharp.png"),
            "deep_prior_hr": _display_path(images_dir / "deep_prior_hr.png"),
            "deep_prior_centered": _display_path(images_dir / "deep_prior_centered.png"),
        },
        "metrics": {
            "bicubic_psnr": _psnr(imgs["HR_np"], imgs["bicubic_np"]),
            "nearest_psnr": _psnr(imgs["HR_np"], imgs["nearest_np"]),
            "final_psnr_lr": psnr_history[-1][0] if psnr_history else None,
            "final_psnr_hr": psnr_history[-1][1] if psnr_history else None,
            "best_psnr_hr": max((x[1] for x in psnr_history), default=None),
        },
        "metrics_csv": _display_path(metrics_path),
        "config": _display_path(config_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run Deep Image Prior super-resolution.")
    parser.add_argument("--input", default="data/sr/zebra_GT.png", help="Input HR image path.")
    parser.add_argument("--factor", type=int, default=4, help="Super-resolution factor, e.g. 2, 4, or 8.")
    parser.add_argument("--output-dir", default=None, help="Output directory.")
    parser.add_argument("--imsize", type=int, default=-1, help="Resize input before running, or -1.")
    parser.add_argument("--enforse-div32", default="CROP", help="Crop image to dimensions divisible by 32.")
    parser.add_argument("--num-iter", type=int, default=None, help="Optimization iterations.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--tv-weight", type=float, default=0.0, help="TV loss weight.")
    parser.add_argument("--reg-noise-std", type=float, default=None, help="Input noise regularization.")
    parser.add_argument("--device", default=None, help="cuda, mps, cpu, or auto when omitted.")
    parser.add_argument("--save-every", type=int, default=100, help="Save progress image every N iterations.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--quiet", action="store_true", help="Disable iteration logging.")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = run_super_resolution(
        input_path=args.input,
        factor=args.factor,
        output_dir=args.output_dir,
        imsize=args.imsize,
        enforse_div32=args.enforse_div32,
        lr=args.lr,
        tv_weight=args.tv_weight,
        num_iter=args.num_iter,
        reg_noise_std=args.reg_noise_std,
        device=args.device,
        save_every=args.save_every,
        seed=args.seed,
        verbose=not args.quiet,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
