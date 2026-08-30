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
    return REPO_ROOT / "result" / datetime.now().strftime("%Y%m%d_%H%M%S") / "restoration"


def save_image_np(path, image_np):
    import numpy as np
    from utils.common_utils import np_to_pil

    path.parent.mkdir(parents=True, exist_ok=True)
    np_to_pil(np.clip(image_np, 0, 1)).save(path)


def run_restoration(
    input_path="data/restoration/barbara.png",
    output_dir=None,
    imsize=-1,
    zero_fraction=None,
    num_iter=None,
    lr=None,
    reg_noise_std=None,
    optimizer="adam",
    pad="reflection",
    input_mode="noise",
    input_depth=32,
    opt_over="net",
    show_every=50,
    save_every=50,
    device=None,
    seed=None,
    verbose=True,
):
    import numpy as np
    import torch
    import torch.nn as nn
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr

    from models import get_net
    from models.skip import skip
    from utils.common_utils import get_image, get_noise, get_params, np_to_pil, pil_to_np, torch_to_np, optimize
    from utils.device_utils import get_device
    from utils.inpainting_utils import get_bernoulli_mask

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    device = get_device() if device is None else torch.device(device)
    input_path = _resolve_path(input_path)
    output_dir = _default_output_dir(output_dir)
    images_dir = output_dir / "images"
    progress_dir = output_dir / "progress"
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    if save_every:
        progress_dir.mkdir(parents=True, exist_ok=True)

    img_pil, img_np = get_image(str(input_path), imsize)
    is_barbara = "barbara" in input_path.name
    is_kate = "kate" in input_path.name

    if is_barbara:
        img_np = nn.ReflectionPad2d(1)(torch.from_numpy(img_np)[None, :])[0].numpy()
        img_pil = np_to_pil(img_np)
        mask_fraction = 0.50 if zero_fraction is None else zero_fraction
        img_mask = get_bernoulli_mask(img_pil, mask_fraction)
        img_mask_np = pil_to_np(img_mask)
        defaults = {"lr": 0.001, "num_iter": 11000, "reg_noise_std": 0.03}
        net = get_net(
            input_depth,
            "skip",
            pad,
            n_channels=1,
            skip_n33d=128,
            skip_n33u=128,
            skip_n11=4,
            num_scales=5,
            upsample_mode="bilinear",
        ).to(device)
    elif is_kate:
        mask_fraction = 0.98 if zero_fraction is None else zero_fraction
        img_mask = get_bernoulli_mask(img_pil, mask_fraction)
        img_mask_np = pil_to_np(img_mask)
        img_mask_np[1] = img_mask_np[0]
        img_mask_np[2] = img_mask_np[0]
        defaults = {"lr": 0.01, "num_iter": 1000, "reg_noise_std": 0.0}
        net = skip(
            input_depth,
            img_np.shape[0],
            num_channels_down=[16, 32, 64, 128, 128],
            num_channels_up=[16, 32, 64, 128, 128],
            num_channels_skip=[0, 0, 0, 0, 0],
            filter_size_down=3,
            filter_size_up=3,
            filter_skip_size=1,
            upsample_mode="bilinear",
            downsample_mode="avg",
            need_sigmoid=True,
            need_bias=True,
            pad=pad,
        ).to(device)
    else:
        raise ValueError("restoration defaults are defined for barbara.png and kate.png")

    lr = defaults["lr"] if lr is None else lr
    num_iter = defaults["num_iter"] if num_iter is None else num_iter
    reg_noise_std = defaults["reg_noise_std"] if reg_noise_std is None else reg_noise_std

    img_masked = img_np * img_mask_np
    save_image_np(images_dir / "input.png", img_np)
    save_image_np(images_dir / "mask.png", img_mask_np)
    save_image_np(images_dir / "masked_input.png", img_masked)

    config = {
        "input_path": _display_path(input_path),
        "zero_fraction": mask_fraction,
        "num_iter": num_iter,
        "lr": lr,
        "reg_noise_std": reg_noise_std,
        "device": str(device),
        "save_every": save_every,
        "seed": seed,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    mse = torch.nn.MSELoss().to(device)
    img_var = torch.from_numpy(img_np)[None, :].to(device)
    mask_var = torch.from_numpy(img_mask_np)[None, :].to(device)
    net_input = get_noise(input_depth, input_mode, img_np.shape[1:]).to(device).detach()
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    state = {"i": 0, "psnr_masked_last": 0, "last_net": None}
    metrics_path = output_dir / "metrics.csv"

    with metrics_path.open("w", newline="") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=["iteration", "loss", "psnr_masked", "psnr"])
        writer.writeheader()

        def closure():
            current_input = net_input_saved
            if reg_noise_std > 0:
                current_input = net_input_saved + (noise.normal_() * reg_noise_std)
            out = net(current_input)
            total_loss = mse(out * mask_var, img_var * mask_var)
            total_loss.backward()
            out_np = torch_to_np(out)
            psnr_masked = float(compare_psnr(img_masked, out_np * img_mask_np, data_range=1.0))
            psnr = float(compare_psnr(img_np, out_np, data_range=1.0))
            loss = float(total_loss.detach().cpu().item())
            i = state["i"]
            writer.writerow({"iteration": i, "loss": loss, "psnr_masked": psnr_masked, "psnr": psnr})
            metrics_file.flush()
            if verbose:
                print("Iteration %05d    Loss %f PSNR_masked %f PSNR %f" % (i, loss, psnr_masked, psnr), "\r", end="")
            if save_every and i % save_every == 0:
                save_image_np(progress_dir / ("iter_%05d.png" % i), out_np)
            if i % show_every == 0:
                if psnr_masked - state["psnr_masked_last"] < -5 and state["last_net"] is not None:
                    for new_param, net_param in zip(state["last_net"], net.parameters()):
                        with torch.no_grad():
                            net_param.copy_(new_param.to(device))
                    return total_loss * 0
                state["last_net"] = [x.detach().cpu() for x in net.parameters()]
                state["psnr_masked_last"] = psnr_masked
            state["i"] += 1
            return total_loss

        optimize(optimizer, get_params(opt_over, net, net_input), closure, LR=lr, num_iter=num_iter)

    if verbose:
        print()
    with torch.no_grad():
        restored_np = np.clip(torch_to_np(net(net_input_saved)), 0, 1)
    save_image_np(images_dir / "restored.png", restored_np)

    summary = {
        "status": "success",
        "output_dir": _display_path(output_dir),
        "images": {
            "input": _display_path(images_dir / "input.png"),
            "mask": _display_path(images_dir / "mask.png"),
            "masked_input": _display_path(images_dir / "masked_input.png"),
            "restored": _display_path(images_dir / "restored.png"),
        },
        "metrics_csv": _display_path(metrics_path),
        "final_psnr": float(compare_psnr(img_np, restored_np, data_range=1.0)),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run Deep Image Prior restoration.")
    parser.add_argument("--input", default="data/restoration/barbara.png")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--zero-fraction", type=float, default=None)
    parser.add_argument("--num-iter", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = run_restoration(
        input_path=args.input,
        output_dir=args.output_dir,
        zero_fraction=args.zero_fraction,
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
