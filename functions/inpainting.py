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
    return REPO_ROOT / "result" / datetime.now().strftime("%Y%m%d_%H%M%S") / "inpainting"


def save_image_np(path, image_np):
    import numpy as np
    from utils.common_utils import np_to_pil

    path.parent.mkdir(parents=True, exist_ok=True)
    np_to_pil(np.clip(image_np, 0, 1)).save(path)


def build_inpainting_net(img_path, img_np, net_type, input_depth, pad, device):
    import torch
    from models.resnet import ResNet
    from models.unet import UNet
    from models.skip import skip

    img_path = str(img_path)
    if "vase.png" in img_path:
        return skip(
            input_depth,
            img_np.shape[0],
            num_channels_down=[128] * 5,
            num_channels_up=[128] * 5,
            num_channels_skip=[0] * 5,
            upsample_mode="nearest",
            filter_skip_size=1,
            filter_size_up=3,
            filter_size_down=3,
            need_sigmoid=True,
            need_bias=True,
            pad=pad,
            act_fun="LeakyReLU",
        ).to(device)

    if ("kate.png" in img_path) or ("peppers.png" in img_path):
        return skip(
            input_depth,
            img_np.shape[0],
            num_channels_down=[128] * 5,
            num_channels_up=[128] * 5,
            num_channels_skip=[128] * 5,
            filter_size_up=3,
            filter_size_down=3,
            upsample_mode="nearest",
            filter_skip_size=1,
            need_sigmoid=True,
            need_bias=True,
            pad=pad,
            act_fun="LeakyReLU",
        ).to(device)

    if "library.png" in img_path and "skip" in net_type:
        depth = int(net_type[-1])
        return skip(
            input_depth,
            img_np.shape[0],
            num_channels_down=[16, 32, 64, 128, 128, 128][:depth],
            num_channels_up=[16, 32, 64, 128, 128, 128][:depth],
            num_channels_skip=[0, 0, 0, 0, 0, 0][:depth],
            filter_size_up=3,
            filter_size_down=5,
            filter_skip_size=1,
            upsample_mode="nearest",
            need1x1_up=False,
            need_sigmoid=True,
            need_bias=True,
            pad=pad,
            act_fun="LeakyReLU",
        ).to(device)

    if "library.png" in img_path and net_type == "UNET":
        return UNet(
            num_input_channels=input_depth,
            num_output_channels=3,
            feature_scale=8,
            more_layers=1,
            concat_x=False,
            upsample_mode="deconv",
            pad="zero",
            norm_layer=torch.nn.InstanceNorm2d,
            need_sigmoid=True,
            need_bias=True,
        ).to(device)

    if "library.png" in img_path and net_type == "ResNet":
        return ResNet(input_depth, img_np.shape[0], 8, 32, need_sigmoid=True, act_fun="LeakyReLU").to(device)

    raise ValueError("Unsupported inpainting image/net combination: %s, %s" % (img_path, net_type))


def default_params_for_image(img_path, net_type):
    img_path = str(img_path)
    if "vase.png" in img_path:
        return {"input_mode": "meshgrid", "input_depth": 2, "lr": 0.01, "num_iter": 5001, "reg_noise_std": 0.03, "param_noise": False, "show_every": 50}
    if ("kate.png" in img_path) or ("peppers.png" in img_path):
        return {"input_mode": "noise", "input_depth": 32, "lr": 0.01, "num_iter": 6001, "reg_noise_std": 0.03, "param_noise": False, "show_every": 50}
    if "library.png" in img_path:
        return {"input_mode": "noise", "input_depth": 1, "lr": 0.001 if net_type in ["UNET", "ResNet"] else 0.01, "num_iter": 3001, "reg_noise_std": 0.0, "param_noise": net_type.startswith("skip"), "show_every": 50}
    return {"input_mode": "noise", "input_depth": 32, "lr": 0.01, "num_iter": 3000, "reg_noise_std": 0.03, "param_noise": False, "show_every": 50}


def run_inpainting(
    input_path="data/inpainting/kate.png",
    mask_path="data/inpainting/kate_mask.png",
    output_dir=None,
    imsize=-1,
    dim_div_by=64,
    net_type="skip_depth6",
    num_iter=None,
    lr=None,
    reg_noise_std=None,
    optimizer="adam",
    pad="reflection",
    opt_over="net",
    device=None,
    save_every=50,
    seed=None,
    verbose=True,
):
    import numpy as np
    import torch

    from utils.common_utils import crop_image, get_image, get_noise, get_params, pil_to_np, torch_to_np, optimize
    from utils.device_utils import get_device

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    device = get_device() if device is None else torch.device(device)
    input_path = _resolve_path(input_path)
    mask_path = _resolve_path(mask_path)
    output_dir = _default_output_dir(output_dir)
    images_dir = output_dir / "images"
    progress_dir = output_dir / "progress"
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    if save_every:
        progress_dir.mkdir(parents=True, exist_ok=True)

    img_pil, _ = get_image(str(input_path), imsize)
    mask_pil, _ = get_image(str(mask_path), imsize)
    img_pil = crop_image(img_pil, dim_div_by)
    mask_pil = crop_image(mask_pil, dim_div_by)
    img_np = pil_to_np(img_pil)
    mask_np = pil_to_np(mask_pil)

    params = default_params_for_image(input_path, net_type)
    if num_iter is not None:
        params["num_iter"] = num_iter
    if lr is not None:
        params["lr"] = lr
    if reg_noise_std is not None:
        params["reg_noise_std"] = reg_noise_std

    save_image_np(images_dir / "input.png", img_np)
    save_image_np(images_dir / "mask.png", mask_np)
    save_image_np(images_dir / "masked_input.png", img_np * mask_np)

    net = build_inpainting_net(input_path, img_np, net_type, params["input_depth"], pad, device)
    net_input = get_noise(params["input_depth"], params["input_mode"], img_np.shape[1:]).to(device).detach()
    mse = torch.nn.MSELoss().to(device)
    img_var = torch.from_numpy(img_np)[None, :].to(device)
    mask_var = torch.from_numpy(mask_np)[None, :].to(device)
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    state = {"i": 0}
    metrics_path = output_dir / "metrics.csv"

    config = {
        "input_path": _display_path(input_path),
        "mask_path": _display_path(mask_path),
        "net_type": net_type,
        "device": str(device),
        "save_every": save_every,
        **params,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    with metrics_path.open("w", newline="") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=["iteration", "loss"])
        writer.writeheader()

        def closure():
            current_input = net_input_saved
            if params["reg_noise_std"] > 0:
                current_input = net_input_saved + (noise.normal_() * params["reg_noise_std"])

            out = net(current_input)
            total_loss = mse(out * mask_var, img_var * mask_var)
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

        optimize(optimizer, get_params(opt_over, net, net_input), closure, params["lr"], params["num_iter"])

    if verbose:
        print()

    with torch.no_grad():
        final_np = np.clip(torch_to_np(net(net_input_saved)), 0, 1)
    save_image_np(images_dir / "inpainted.png", final_np)

    summary = {
        "status": "success",
        "output_dir": _display_path(output_dir),
        "images": {
            "input": _display_path(images_dir / "input.png"),
            "mask": _display_path(images_dir / "mask.png"),
            "masked_input": _display_path(images_dir / "masked_input.png"),
            "inpainted": _display_path(images_dir / "inpainted.png"),
        },
        "metrics_csv": _display_path(metrics_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run Deep Image Prior inpainting.")
    parser.add_argument("--input", default="data/inpainting/kate.png")
    parser.add_argument("--mask", default="data/inpainting/kate_mask.png")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--net-type", default="skip_depth6")
    parser.add_argument("--num-iter", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = run_inpainting(
        input_path=args.input,
        mask_path=args.mask,
        output_dir=args.output_dir,
        net_type=args.net_type,
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
