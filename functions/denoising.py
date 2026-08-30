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
    return REPO_ROOT / "result" / datetime.now().strftime("%Y%m%d_%H%M%S") / "denoising"


def save_image_np(path, image_np):
    import numpy as np
    from utils.common_utils import np_to_pil

    path.parent.mkdir(parents=True, exist_ok=True)
    np_to_pil(np.clip(image_np, 0, 1)).save(path)


def run_denoising(
    input_path="data/denoising/F16_GT.png",
    output_dir=None,
    imsize=-1,
    sigma=25,
    num_iter=None,
    lr=0.01,
    reg_noise_std=1.0 / 30.0,
    optimizer="adam",
    input_mode="noise",
    pad="reflection",
    opt_over="net",
    exp_weight=0.99,
    show_every=100,
    save_every=100,
    device=None,
    seed=None,
    verbose=True,
):
    import numpy as np
    import torch
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr

    from models import get_net
    from models.skip import skip
    from utils.common_utils import (
        crop_image,
        get_image,
        get_noise,
        get_params,
        pil_to_np,
        torch_to_np,
        optimize,
    )
    from utils.denoising_utils import get_noisy_image
    from utils.device_utils import get_device

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

    sigma_scaled = sigma / 255.0
    fname = str(input_path)
    is_snail = input_path.name == "snail.jpg"
    is_f16 = input_path.name == "F16_GT.png"

    if is_snail:
        img_noisy_pil = crop_image(get_image(fname, imsize)[0], d=32)
        img_noisy_np = pil_to_np(img_noisy_pil)
        img_pil = img_noisy_pil
        img_np = img_noisy_np
        default_num_iter = 2400
        input_depth = 3
        net = skip(
            input_depth,
            3,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode="bilinear",
            need_sigmoid=True,
            need_bias=True,
            pad=pad,
            act_fun="LeakyReLU",
        ).to(device)
    else:
        img_pil = crop_image(get_image(fname, imsize)[0], d=32)
        img_np = pil_to_np(img_pil)
        img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_scaled)
        default_num_iter = 3000 if is_f16 else 3000
        input_depth = 32
        net = get_net(
            input_depth,
            "skip",
            pad,
            skip_n33d=128,
            skip_n33u=128,
            skip_n11=4,
            num_scales=5,
            upsample_mode="bilinear",
        ).to(device)

    if num_iter is None:
        num_iter = default_num_iter

    save_image_np(images_dir / "input.png", img_np)
    save_image_np(images_dir / "noisy.png", img_noisy_np)

    config = {
        "input_path": _display_path(input_path),
        "imsize": imsize,
        "sigma": sigma,
        "num_iter": num_iter,
        "lr": lr,
        "reg_noise_std": reg_noise_std,
        "optimizer": optimizer,
        "device": str(device),
        "save_every": save_every,
        "seed": seed,
        "output_dir": _display_path(output_dir),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    net_input = get_noise(input_depth, input_mode, (img_pil.size[1], img_pil.size[0])).to(device).detach()
    mse = torch.nn.MSELoss().to(device)
    img_noisy_torch = torch.from_numpy(img_noisy_np)[None, :].to(device)

    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    state = {"i": 0, "out_avg": None, "psnr_noisy_last": 0, "last_net": None}
    metrics_path = output_dir / "metrics.csv"

    with metrics_path.open("w", newline="") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=["iteration", "loss", "psnr_noisy", "psnr_gt", "psnr_gt_sm"])
        writer.writeheader()

        def closure():
            if reg_noise_std > 0:
                current_input = net_input_saved + (noise.normal_() * reg_noise_std)
            else:
                current_input = net_input_saved

            out = net(current_input)
            if state["out_avg"] is None:
                state["out_avg"] = out.detach()
            else:
                state["out_avg"] = state["out_avg"] * exp_weight + out.detach() * (1 - exp_weight)

            total_loss = mse(out, img_noisy_torch)
            total_loss.backward()

            out_np = torch_to_np(out)
            avg_np = torch_to_np(state["out_avg"])
            psnr_noisy = float(compare_psnr(img_noisy_np, out_np, data_range=1.0))
            psnr_gt = float(compare_psnr(img_np, out_np, data_range=1.0))
            psnr_gt_sm = float(compare_psnr(img_np, avg_np, data_range=1.0))
            loss = float(total_loss.detach().cpu().item())
            i = state["i"]

            writer.writerow({"iteration": i, "loss": loss, "psnr_noisy": psnr_noisy, "psnr_gt": psnr_gt, "psnr_gt_sm": psnr_gt_sm})
            metrics_file.flush()

            if verbose:
                print("Iteration %05d    Loss %f   PSNR_noisy %f   PSNR_gt %f" % (i, loss, psnr_noisy, psnr_gt), "\r", end="")

            if save_every and i % save_every == 0:
                save_image_np(progress_dir / ("iter_%05d.png" % i), out_np)

            if i % show_every:
                if psnr_noisy - state["psnr_noisy_last"] < -5 and state["last_net"] is not None:
                    for new_param, net_param in zip(state["last_net"], net.parameters()):
                        with torch.no_grad():
                            net_param.copy_(new_param.to(device))
                    return total_loss * 0
                state["last_net"] = [x.detach().cpu() for x in net.parameters()]
                state["psnr_noisy_last"] = psnr_noisy

            state["i"] += 1
            return total_loss

        optimize(optimizer, get_params(opt_over, net, net_input), closure, lr, num_iter)

    if verbose:
        print()

    with torch.no_grad():
        final_np = np.clip(torch_to_np(net(net_input_saved)), 0, 1)
    save_image_np(images_dir / "denoised.png", final_np)

    summary = {
        "status": "success",
        "output_dir": _display_path(output_dir),
        "images": {
            "input": _display_path(images_dir / "input.png"),
            "noisy": _display_path(images_dir / "noisy.png"),
            "denoised": _display_path(images_dir / "denoised.png"),
        },
        "metrics_csv": _display_path(metrics_path),
        "final_psnr": float(compare_psnr(img_np, final_np, data_range=1.0)),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run Deep Image Prior denoising.")
    parser.add_argument("--input", default="data/denoising/F16_GT.png")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--sigma", type=float, default=25)
    parser.add_argument("--num-iter", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = run_denoising(
        input_path=args.input,
        output_dir=args.output_dir,
        sigma=args.sigma,
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
