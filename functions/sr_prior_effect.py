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
    return path if path.is_absolute() else REPO_ROOT / path


def _display_path(path):
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _default_output_dir(output_dir):
    if output_dir is not None:
        return _resolve_path(output_dir)
    return REPO_ROOT / "result" / datetime.now().strftime("%Y%m%d_%H%M%S") / "sr_prior_effect"


def save_image_np(path, image_np):
    import numpy as np
    from utils.common_utils import np_to_pil

    path.parent.mkdir(parents=True, exist_ok=True)
    np_to_pil(np.clip(image_np, 0, 1)).save(path)


def run_sr_prior_effect(
    input_path="data/sr/zebra_crop.png",
    factor=4,
    output_dir=None,
    imsize=-1,
    enforse_div32="CROP",
    num_iter=2000,
    lr=0.01,
    branches=("direct", "tv", "deep"),
    device=None,
    save_every=500,
    seed=None,
    verbose=True,
):
    import numpy as np
    import torch
    import torch.nn as nn
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr

    from models.downsampler import Downsampler
    from models.skip import skip
    from utils.common_utils import get_noise, get_params, np_to_torch, torch_to_np, optimize
    from utils.device_utils import get_device
    from utils.sr_utils import get_baselines, load_LR_HR_imgs_sr, put_in_center, tv_loss

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    device = get_device() if device is None else torch.device(device)
    input_path = _resolve_path(input_path)
    output_dir = _default_output_dir(output_dir)
    images_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    imgs = load_LR_HR_imgs_sr(str(input_path), imsize, factor, enforse_div32)
    imgs["bicubic_np"], imgs["sharp_np"], imgs["nearest_np"] = get_baselines(imgs["LR_pil"], imgs["HR_pil"])
    save_image_np(images_dir / "input_hr.png", imgs["HR_np"])
    save_image_np(images_dir / "input_lr.png", imgs["LR_np"])
    save_image_np(images_dir / "bicubic.png", imgs["bicubic_np"])

    config = {
        "input_path": _display_path(input_path),
        "factor": factor,
        "num_iter": num_iter,
        "lr": lr,
        "branches": list(branches),
        "device": str(device),
        "save_every": save_every,
        "seed": seed,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    input_depth = 3
    input_mode = "noise"
    pad = "reflection"
    kernel_type = "lanczos2"
    img_LR_var = np_to_torch(imgs["LR_np"]).to(device)
    mse = torch.nn.MSELoss().to(device)

    def run_branch(name, net, net_input, opt_over, tv_weight, reg_noise_std):
        branch_dir = output_dir / name
        branch_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = branch_dir / "metrics.csv"
        downsampler = Downsampler(n_planes=3, factor=factor, kernel_type=kernel_type, phase=0.5, preserve_size=True).to(device)
        net_input_saved = net_input.detach().clone()
        noise = net_input.detach().clone()
        history = []
        state = {"i": 0}

        with metrics_path.open("w", newline="") as metrics_file:
            writer = csv.DictWriter(metrics_file, fieldnames=["iteration", "psnr_lr", "psnr_hr", "loss"])
            writer.writeheader()

            def closure():
                current_input = net_input_saved
                if reg_noise_std > 0:
                    current_input = net_input_saved + (noise.normal_() * reg_noise_std)
                out_HR = net(current_input)
                out_LR = downsampler(out_HR)
                total_loss = mse(out_LR, img_LR_var) + tv_weight * tv_loss(out_HR)
                total_loss.backward()
                out_LR_np = torch_to_np(out_LR)
                out_HR_np = torch_to_np(out_HR)
                psnr_LR = float(compare_psnr(imgs["LR_np"], out_LR_np, data_range=1.0))
                psnr_HR = float(compare_psnr(imgs["HR_np"], out_HR_np, data_range=1.0))
                loss = float(total_loss.detach().cpu().item())
                i = state["i"]
                writer.writerow({"iteration": i, "psnr_lr": psnr_LR, "psnr_hr": psnr_HR, "loss": loss})
                metrics_file.flush()
                history.append([psnr_LR, psnr_HR])
                if verbose:
                    print("%s %05d    PSNR_LR %.3f   PSNR_HR %.3f" % (name, i, psnr_LR, psnr_HR), "\r", end="")
                if save_every and i % save_every == 0:
                    save_image_np(branch_dir / ("iter_%05d.png" % i), out_HR_np)
                state["i"] += 1
                return total_loss

            optimize("adam", get_params(opt_over, net, net_input), closure, lr, num_iter)

        with torch.no_grad():
            out_np = np.clip(torch_to_np(net(net_input_saved)), 0, 1)
        centered = put_in_center(out_np, imgs["orig_np"].shape[1:])
        save_image_np(images_dir / ("%s.png" % name), centered)
        return {"image": _display_path(images_dir / ("%s.png" % name)), "metrics_csv": _display_path(metrics_path), "best_psnr_hr": max((x[1] for x in history), default=None)}

    results = {}
    hr_size = (imgs["HR_pil"].size[1], imgs["HR_pil"].size[0])

    if "direct" in branches:
        net = nn.Sequential()
        net_input = get_noise(input_depth, input_mode, hr_size).to(device).detach()
        results["direct"] = run_branch("direct", net, net_input, "input", tv_weight=0.0, reg_noise_std=0.0)

    if "tv" in branches:
        net = nn.Sequential()
        net_input = get_noise(input_depth, input_mode, hr_size).to(device).detach()
        results["tv"] = run_branch("tv", net, net_input, "input", tv_weight=1e-7, reg_noise_std=0.0)

    if "deep" in branches:
        net = skip(
            input_depth,
            3,
            num_channels_down=[128, 128, 128, 128, 128],
            num_channels_up=[128, 128, 128, 128, 128],
            num_channels_skip=[4, 4, 4, 4, 4],
            upsample_mode="bilinear",
            need_sigmoid=True,
            need_bias=True,
            pad=pad,
            act_fun="LeakyReLU",
        ).to(device)
        net_input = get_noise(input_depth, input_mode, hr_size).to(device).detach()
        results["deep"] = run_branch("deep", net, net_input, "net", tv_weight=0.0, reg_noise_std=1.0 / 30.0)

    if verbose:
        print()
    summary = {
        "status": "success",
        "output_dir": _display_path(output_dir),
        "results": results,
        "baselines": {
            "bicubic_psnr": float(compare_psnr(imgs["HR_np"], imgs["bicubic_np"], data_range=1.0)),
            "nearest_psnr": float(compare_psnr(imgs["HR_np"], imgs["nearest_np"], data_range=1.0)),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run Deep Image Prior SR prior effect experiment.")
    parser.add_argument("--input", default="data/sr/zebra_crop.png")
    parser.add_argument("--factor", type=int, default=4)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-iter", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--branches", default="direct,tv,deep")
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = run_sr_prior_effect(
        input_path=args.input,
        factor=args.factor,
        output_dir=args.output_dir,
        num_iter=args.num_iter,
        lr=args.lr,
        branches=tuple(x.strip() for x in args.branches.split(",") if x.strip()),
        device=args.device,
        save_every=args.save_every,
        seed=args.seed,
        verbose=not args.quiet,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
