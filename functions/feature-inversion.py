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
    return REPO_ROOT / "result" / datetime.now().strftime("%Y%m%d_%H%M%S") / "feature_inversion"


def save_image_np(path, image_np):
    import numpy as np
    from utils.common_utils import np_to_pil

    path.parent.mkdir(parents=True, exist_ok=True)
    np_to_pil(np.clip(image_np, 0, 1)).save(path)


def run_feature_inversion(
    input_path="data/feature_inversion/building.jpg",
    output_dir=None,
    pretrained_net="alexnet_caffe",
    layers_to_use="fc6",
    num_iter=3100,
    lr=0.001,
    input_depth=32,
    imsize_net=256,
    pad="zero",
    optimizer="adam",
    opt_over="net",
    save_every=200,
    device=None,
    seed=None,
    verbose=True,
):
    import numpy as np
    import torch

    from models.skip import skip
    from utils.common_utils import get_image, get_noise, get_params, torch_to_np, optimize
    from utils.device_utils import get_device
    from utils.feature_inversion_utils import get_deprocessor, get_matcher, get_preprocessor, vgg_preprocess_var
    from utils.perceptual_loss.perceptual_loss import get_pretrained_net

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

    cnn = get_pretrained_net(pretrained_net).to(device)
    opt_content = {"layers": layers_to_use, "what": "features"}
    keys = [x for x in cnn._modules.keys()]
    max_idx = max(keys.index(x) for x in opt_content["layers"].split(","))
    for k in keys[max_idx + 1:]:
        cnn._modules.pop(k)
    cnn.eval()

    imsize = 227 if pretrained_net == "alexnet_caffe" else 224
    preprocess, _ = get_preprocessor(imsize), get_deprocessor()
    img_content_pil, img_content_np = get_image(str(input_path), imsize)
    img_content_preprocessed = preprocess(img_content_pil)[None, :].to(device)
    save_image_np(images_dir / "content.png", img_content_np)

    matcher_content = get_matcher(cnn, opt_content)
    matcher_content.mode = "store"
    cnn(img_content_preprocessed)

    net_input = get_noise(input_depth, "noise", imsize_net).to(device).detach()
    net = skip(
        input_depth,
        3,
        num_channels_down=[16, 32, 64, 128, 128, 128],
        num_channels_up=[16, 32, 64, 128, 128, 128],
        num_channels_skip=[4, 4, 4, 4, 4, 4],
        filter_size_down=[7, 7, 5, 5, 3, 3],
        filter_size_up=[7, 7, 5, 5, 3, 3],
        upsample_mode="nearest",
        downsample_mode="avg",
        need_sigmoid=True,
        pad=pad,
        act_fun="LeakyReLU",
    ).to(device)

    config = {
        "input_path": _display_path(input_path),
        "pretrained_net": pretrained_net,
        "layers_to_use": layers_to_use,
        "num_iter": num_iter,
        "lr": lr,
        "device": str(device),
        "save_every": save_every,
        "seed": seed,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    metrics_path = output_dir / "metrics.csv"
    state = {"i": 0}
    matcher_content.mode = "match"

    with metrics_path.open("w", newline="") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=["iteration", "loss"])
        writer.writeheader()

        def closure():
            out = net(net_input)[:, :, :imsize, :imsize]
            cnn(vgg_preprocess_var(out))
            total_loss = sum(matcher_content.losses.values())
            total_loss.backward()
            loss = float(total_loss.detach().cpu().item())
            i = state["i"]
            writer.writerow({"iteration": i, "loss": loss})
            metrics_file.flush()
            if verbose:
                print("Iteration %05d    Loss %.3f" % (i, loss), "\r", end="")
            if save_every and i % save_every == 0:
                save_image_np(progress_dir / ("iter_%05d.png" % i), torch_to_np(out))
            state["i"] += 1
            return total_loss

        optimize(optimizer, get_params(opt_over, net, net_input), closure, lr, num_iter)

    if verbose:
        print()
    with torch.no_grad():
        out_np = np.clip(torch_to_np(net(net_input)[:, :, :imsize, :imsize]), 0, 1)
    save_image_np(images_dir / "feature_inversion.png", out_np)

    summary = {
        "status": "success",
        "output_dir": _display_path(output_dir),
        "images": {"content": _display_path(images_dir / "content.png"), "feature_inversion": _display_path(images_dir / "feature_inversion.png")},
        "metrics_csv": _display_path(metrics_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run Deep Image Prior feature inversion.")
    parser.add_argument("--input", default="data/feature_inversion/building.jpg")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--pretrained-net", default="alexnet_caffe")
    parser.add_argument("--layers", default="fc6")
    parser.add_argument("--num-iter", type=int, default=3100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = run_feature_inversion(
        input_path=args.input,
        output_dir=args.output_dir,
        pretrained_net=args.pretrained_net,
        layers_to_use=args.layers,
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
