import argparse
import base64
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

FUNCTION_NOTEBOOKS = {
    "activation_maximization": "activation_maximization.ipynb",
    "denoising": "denoising.ipynb",
    "feature_inversion": "feature_inversion.ipynb",
    "flash_no_flash": "flash-no-flash.ipynb",
    "inpainting": "inpainting.ipynb",
    "restoration": "restoration.ipynb",
    "sr_prior_effect": "sr_prior_effect.ipynb",
    "super_resolution": "super-resolution.ipynb",
}

ALIASES = {
    "activation-maximization": "activation_maximization",
    "flash-no-flash": "flash_no_flash",
    "sr-prior-effect": "sr_prior_effect",
    "super-resolution": "super_resolution",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Deep Image Prior notebooks from the command line."
    )
    parser.add_argument(
        "--function",
        required=False,
        default=None,
        help="Function to run, e.g. restoration, denoising, super_resolution, or all.",
    )
    parser.add_argument(
        "--list-functions",
        action="store_true",
        help="List available functions and exit.",
    )
    parser.add_argument(
        "--output-root",
        default="result",
        help="Directory where timestamped run outputs are written.",
    )
    parser.add_argument(
        "--timestamp",
        default=None,
        help="Optional timestamp directory name. Defaults to current local time.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=-1,
        help="Per-cell timeout in seconds. Use -1 to disable timeout.",
    )
    parser.add_argument(
        "--kernel",
        default="python3",
        help="Jupyter kernel name to use.",
    )
    parser.add_argument(
        "--allow-errors",
        action="store_true",
        help="Keep executing a notebook after cell errors.",
    )
    return parser.parse_args()


def normalize_function(name):
    normalized = name.strip()
    normalized = ALIASES.get(normalized, normalized)
    if normalized == "all":
        return normalized
    if normalized not in FUNCTION_NOTEBOOKS:
        available = ", ".join(["all"] + sorted(FUNCTION_NOTEBOOKS))
        raise ValueError("Unknown function '%s'. Available: %s" % (name, available))
    return normalized


def list_functions():
    for name in sorted(FUNCTION_NOTEBOOKS):
        print("%-24s %s" % (name, FUNCTION_NOTEBOOKS[name]))


def build_run_dir(output_root, timestamp, function_name):
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root_path = Path(output_root)
    if not output_root_path.is_absolute():
        output_root_path = REPO_ROOT / output_root_path
    return output_root_path / timestamp / function_name


def display_path(path):
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def make_setup_cell(run_dir, function_name):
    import nbformat

    source = f"""
# Injected by main.py. Notebook code can use RESULT_DIR when it wants
# to write additional artifacts next to the executed notebook.
from pathlib import Path
import os

RESULT_DIR = Path(r"{run_dir}")
ARTIFACTS_DIR = RESULT_DIR / "artifacts"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
os.environ["DIP_RESULT_DIR"] = str(RESULT_DIR)
os.environ["DIP_FUNCTION"] = "{function_name}"
"""
    return nbformat.v4.new_code_cell(source.strip() + "\n")


def extract_artifacts(nb, artifacts_dir):
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for cell_index, cell in enumerate(nb.cells):
        for output_index, output in enumerate(cell.get("outputs", [])):
            data = output.get("data", {})
            if "image/png" not in data:
                continue

            raw_png = data["image/png"]
            if isinstance(raw_png, list):
                raw_png = "".join(raw_png)

            filename = "cell_%03d_output_%03d.png" % (cell_index, output_index)
            path = artifacts_dir / filename
            path.write_bytes(base64.b64decode(raw_png))
            written.append(display_path(path))

    return written


def write_run_info(path, info):
    path.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")


def execute_notebook(function_name, notebook_name, args, timestamp):
    import nbformat
    from nbclient import NotebookClient

    notebook_path = REPO_ROOT / notebook_name
    if not notebook_path.exists():
        raise FileNotFoundError("Notebook not found: %s" % notebook_path)

    run_dir = build_run_dir(args.output_root, timestamp, function_name)
    artifacts_dir = run_dir / "artifacts"
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    executed_notebook = run_dir / "executed_notebook.ipynb"
    run_info_path = run_dir / "run.json"

    started_at = datetime.now().isoformat(timespec="seconds")
    info = {
        "function": function_name,
        "notebook": notebook_name,
        "started_at": started_at,
        "status": "running",
        "output_dir": display_path(run_dir),
        "artifacts_dir": display_path(artifacts_dir),
        "kernel": args.kernel,
        "timeout": args.timeout,
    }
    write_run_info(run_info_path, info)

    nb = nbformat.read(notebook_path, as_version=4)
    nb.cells.insert(0, make_setup_cell(run_dir, function_name))

    old_env = {
        "DIP_RESULT_DIR": os.environ.get("DIP_RESULT_DIR"),
        "DIP_FUNCTION": os.environ.get("DIP_FUNCTION"),
    }
    os.environ["DIP_RESULT_DIR"] = str(run_dir)
    os.environ["DIP_FUNCTION"] = function_name

    try:
        client = NotebookClient(
            nb,
            timeout=args.timeout,
            kernel_name=args.kernel,
            allow_errors=args.allow_errors,
            resources={"metadata": {"path": str(REPO_ROOT)}},
        )
        client.execute()
        artifacts = extract_artifacts(nb, artifacts_dir)
        nbformat.write(nb, executed_notebook)

        info.update(
            {
                "status": "success",
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "executed_notebook": display_path(executed_notebook),
                "artifacts": artifacts,
            }
        )
        write_run_info(run_info_path, info)
        return True, info
    except Exception as exc:
        nbformat.write(nb, executed_notebook)
        info.update(
            {
                "status": "failed",
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "executed_notebook": display_path(executed_notebook),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
        write_run_info(run_info_path, info)
        return False, info
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main():
    args = parse_args()

    if args.list_functions:
        list_functions()
        return 0

    if args.function is None:
        print("error: --function is required unless --list-functions is used", file=sys.stderr)
        return 2

    try:
        function_name = normalize_function(args.function)
    except ValueError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2

    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    selected = (
        sorted(FUNCTION_NOTEBOOKS)
        if function_name == "all"
        else [function_name]
    )

    all_ok = True
    for name in selected:
        notebook = FUNCTION_NOTEBOOKS[name]
        print("Running %s (%s)" % (name, notebook))
        ok, info = execute_notebook(name, notebook, args, timestamp)
        all_ok = all_ok and ok
        print("%s: %s -> %s" % (name, info["status"], info["output_dir"]))
        if not ok and not args.allow_errors:
            break

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
