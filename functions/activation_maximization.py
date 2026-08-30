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
    return REPO_ROOT / "result" / datetime.now().strftime("%Y%m%d_%H%M%S") / "activation_maximization"


def save_image_np(path, image_np):
    import numpy as np
    from utils.common_utils import np_to_pil

    path.parent.mkdir(parents=True, exist_ok=True)
    np_to_pil(np.clip(image_np, 0, 1)).save(path)


def find_imagenet_class_index(name):
    mapping_path = REPO_ROOT / "data" / "imagenet1000_clsid_to_human.txt"
    with mapping_path.open("r") as f:
        corresp = json.load(f)
    for key, value in corresp.items():
        if name in value:
            return int(key)
    raise ValueError("Could not find ImageNet class containing: %s" % name)


def run_activation_maximization(
    input_path="data/feature_inversion/building.jpg",
    output_dir=None,
    pretrained_net="alexnet_caffe",
    layer="conv4",
    map_idx=2,
    class_name=None,
    num_iter=3100,
    lr=None,
    input_depth=32,
    imsize_net=256,
    tv_weight=0.0,
    reg_noise_std=0.03,
    param_noise=True,
    window_size=20,
    save_every=100,
    device=None,
    seed=None,
    verbose=True,
):
    import numpy as np
    import torch
    import torch.nn as nn

    from models.skip import skip
    from utils.common_utils import get_image, get_noise, get_params, torch_to_np, optimize
    from utils.device_utils import get_device
    from utils.perceptual_loss.perceptual_loss import get_matcher, get_preprocessor, get_pretrained_net, vgg_preprocess_caffe
    from utils.sr_utils import tv_loss as sr_tv_loss

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    if layer == "fc8" and class_name is not None:
        map_idx = find_imagenet_class_index(class_name)
    if lr is None:
        lr = 0.01 if layer == "fc8" else 0.001

    device = get_device() if device is None else torch.device(device)
    input_path = _resolve_path(input_path)
    output_dir = _default_output_dir(output_dir)
    images_dir = output_dir / "images"
    progress_dir = output_dir / "progress"
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    if save_every:
        progress_dir.mkdir(parents=True, exist_ok=True)

    imsize = 227 if pretrained_net == "alexnet_caffe" else 224
    preprocess = get_preprocessor(imsize)
    img_content_pil, img_content_np = get_image(str(input_path), -1)
    img_content_preprocessed = preprocess(img_content_pil)[None, :].to(device)
    save_image_np(images_dir / "reference.png", img_content_np)

    opt_content = {"layers": [layer], "what": "features", "map_idx": map_idx}
    cnn = get_pretrained_net(pretrained_net).to(device)
    cnn.add_module("softmax", nn.Softmax(dim=1))
    keys = [x for x in cnn._modules.keys()]
    max_idx = max(keys.index(x) for x in opt_content["layers"])
    for k in keys[max_idx + 1:]:
        cnn._modules.pop(k)
    cnn.eval()

    matcher_content = get_matcher(cnn, opt_content)
    matcher_content.mode = "match"
    if layer != "fc8":
        matcher_content.window_size = window_size
        matcher_content.method = "maximize"

    net_input = get_noise(input_depth, "noise", imsize_net).to(device).detach()
    net = skip(
        input_depth,
        3,
        num_channels_down=[16, 32, 64, 128, 128, 128],
        num_channels_up=[16, 32, 64, 128, 128, 128],
        num_channels_skip=[0, 4, 4, 4, 4, 4],
        filter_size_down=[5, 3, 5, 5, 3, 5],
        filter_size_up=[5, 3, 5, 3, 5, 3],
        upsample_mode="bilinear",
        downsample_mode="avg",
        need_sigmoid=True,
        pad="reflection",
        act_fun="LeakyReLU",
    ).to(device)

    config = {
        "input_path": _display_path(input_path),
        "pretrained_net": pretrained_net,
        "layer": layer,
        "map_idx": map_idx,
        "class_name": class_name,
        "num_iter": num_iter,
        "lr": lr,
        "device": str(device),
        "save_every": save_every,
        "seed": seed,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    state = {"i": 0}
    metrics_path = output_dir / "metrics.csv"

    with metrics_path.open("w", newline="") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=["iteration", "loss"])
        writer.writeheader()

        def closure():
            if param_noise:
                for n in [x for x in net.parameters() if len(x.size()) == 4]:
                    n = n + n.detach().clone().normal_() * n.std() / 50
            current_input = net_input_saved
            if reg_noise_std > 0:
                current_input = net_input_saved + (noise.normal_() * reg_noise_std)
            out = net(current_input)[:, :, :imsize, :imsize]
            cnn(vgg_preprocess_caffe(out))
            total_loss = sum(matcher_content.losses.values()) * 5
            if tv_weight > 0:
                total_loss = total_loss + tv_weight * sr_tv_loss(vgg_preprocess_caffe(out), beta=2)
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

        optimize("adam", get_params("net", net, net_input), closure, lr, num_iter)

    if verbose:
        print()
    with torch.no_grad():
        out_np = np.clip(torch_to_np(net(net_input_saved)[:, :, :imsize, :imsize]), 0, 1)
    save_image_np(images_dir / "activation_maximization.png", out_np)

    summary = {
        "status": "success",
        "output_dir": _display_path(output_dir),
        "images": {
            "reference": _display_path(images_dir / "reference.png"),
            "activation_maximization": _display_path(images_dir / "activation_maximization.png"),
        },
        "metrics_csv": _display_path(metrics_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run Deep Image Prior activation maximization.")
    parser.add_argument("--input", default="data/feature_inversion/building.jpg")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--pretrained-net", default="alexnet_caffe")
    parser.add_argument("--layer", default="conv4")
    parser.add_argument("--map-idx", type=int, default=2)
    parser.add_argument("--class-name", default=None)
    parser.add_argument("--num-iter", type=int, default=3100)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = run_activation_maximization(
        input_path=args.input,
        output_dir=args.output_dir,
        pretrained_net=args.pretrained_net,
        layer=args.layer,
        map_idx=args.map_idx,
        class_name=args.class_name,
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
