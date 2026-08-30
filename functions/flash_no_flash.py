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
    return REPO_ROOT / "result" / datetime.now().strftime("%Y%m%d_%H%M%S") / "flash_no_flash"


def save_image_np(path, image_np):
    import numpy as np
    from utils.common_utils import np_to_pil

    path.parent.mkdir(parents=True, exist_ok=True)
    np_to_pil(np.clip(image_np, 0, 1)).save(path)


def run_flash_no_flash(
    flash_path="data/flash_no_flash/cave01_00_flash.jpg",
    noflash_path="data/flash_no_flash/cave01_01_noflash.jpg",
    output_dir=None,
    imsize=-1,
    num_iter=601,
    lr=0.1,
    reg_noise_std=0.0,
    optimizer="adam",
    pad="reflection",
    opt_over="net",
    save_every=50,
    device=None,
    seed=None,
    verbose=True,
):
    import numpy as np
    import torch

    from models.skip import skip
    from utils.common_utils import get_params, np_to_torch, pil_to_np, torch_to_np, optimize
    from utils.device_utils import get_device
    from utils.sr_utils import load_LR_HR_imgs_sr

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    device = get_device() if device is None else torch.device(device)
    flash_path = _resolve_path(flash_path)
    noflash_path = _resolve_path(noflash_path)
    output_dir = _default_output_dir(output_dir)
    images_dir = output_dir / "images"
    progress_dir = output_dir / "progress"
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    if save_every:
        progress_dir.mkdir(parents=True, exist_ok=True)

    img_flash = load_LR_HR_imgs_sr(str(flash_path), imsize, 1, enforse_div32="CROP")["HR_pil"]
    img_noflash = load_LR_HR_imgs_sr(str(noflash_path), imsize, 1, enforse_div32="CROP")["HR_pil"]
    img_flash_np = pil_to_np(img_flash)
    img_noflash_np = pil_to_np(img_noflash)

    save_image_np(images_dir / "flash.png", img_flash_np)
    save_image_np(images_dir / "noflash_target.png", img_noflash_np)

    config = {
        "flash_path": _display_path(flash_path),
        "noflash_path": _display_path(noflash_path),
        "num_iter": num_iter,
        "lr": lr,
        "reg_noise_std": reg_noise_std,
        "device": str(device),
        "save_every": save_every,
        "seed": seed,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    input_depth = 3
    net_input = np_to_torch(img_flash_np).to(device)
    net = skip(
        input_depth,
        3,
        num_channels_down=[128, 128, 128, 128, 128],
        num_channels_up=[128, 128, 128, 128, 128],
        num_channels_skip=[4, 4, 4, 4, 4],
        upsample_mode=["nearest", "nearest", "bilinear", "bilinear", "bilinear"],
        need_sigmoid=True,
        need_bias=True,
        pad=pad,
    ).to(device)
    mse = torch.nn.MSELoss().to(device)
    img_noflash_var = np_to_torch(img_noflash_np).to(device)
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    state = {"i": 0}
    metrics_path = output_dir / "metrics.csv"

    with metrics_path.open("w", newline="") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=["iteration", "loss"])
        writer.writeheader()

        def closure():
            current_input = net_input_saved
            if reg_noise_std > 0:
                current_input = net_input_saved + (noise.normal_() * reg_noise_std)
            out = net(current_input)
            total_loss = mse(out, img_noflash_var)
            total_loss.backward()
            loss = float(total_loss.detach().cpu().item())
            i = state["i"]
            writer.writerow({"iteration": i, "loss": loss})
            metrics_file.flush()
            if verbose:
                print("Iteration %05d    Loss %f" % (i, loss), "\r", end="")
            if save_every and i % save_every == 0:
                save_image_np(progress_dir / ("iter_%05d.png" % i), torch_to_np(out))
            state["i"] += 1
            return total_loss

        optimize(optimizer, get_params(opt_over, net, net_input), closure, lr, num_iter)

    if verbose:
        print()
    with torch.no_grad():
        output_np = np.clip(torch_to_np(net(net_input_saved)), 0, 1)
    save_image_np(images_dir / "reconstructed_noflash.png", output_np)

    summary = {
        "status": "success",
        "output_dir": _display_path(output_dir),
        "images": {
            "flash": _display_path(images_dir / "flash.png"),
            "noflash_target": _display_path(images_dir / "noflash_target.png"),
            "reconstructed_noflash": _display_path(images_dir / "reconstructed_noflash.png"),
        },
        "metrics_csv": _display_path(metrics_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run Deep Image Prior flash/no-flash reconstruction.")
    parser.add_argument("--flash", default="data/flash_no_flash/cave01_00_flash.jpg")
    parser.add_argument("--noflash", default="data/flash_no_flash/cave01_01_noflash.jpg")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-iter", type=int, default=601)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = run_flash_no_flash(
        flash_path=args.flash,
        noflash_path=args.noflash,
        output_dir=args.output_dir,
        num_iter=args.num_iter,
        lr=args.lr,
        device=args.device,
        save_every=args.save_every,
        seed=args.seed,
        verbose=not args.quiet,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
