import argparse
import base64
import ast
import json
import os
import re
import subprocess
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

FUNCTION_SCRIPTS = {
    "activation_maximization": "functions/activation-maximization.py",
    "denoising": "functions/denoising.py",
    "feature_inversion": "functions/feature-inversion.py",
    "flash_no_flash": "functions/flash-no-flash.py",
    "inpainting": "functions/inpainting.py",
    "restoration": "functions/restoration.py",
    "sr_prior_effect": "functions/sr-prior-effect.py",
    "super_resolution": "functions/super-resolution.py",
}

ALIASES = {
    "activation-maximization": "activation_maximization",
    "flash-no-flash": "flash_no_flash",
    "sr-prior-effect": "sr_prior_effect",
    "super-resolution": "super_resolution",
}

INPUT_PARAM_BY_FUNCTION = {
    "activation_maximization": "fname",
    "denoising": "fname",
    "feature_inversion": "fname",
    "inpainting": "img_path",
    "restoration": "f",
    "sr_prior_effect": "fname",
    "super_resolution": "path_to_image",
}

MASK_PARAM_BY_FUNCTION = {
    "inpainting": "mask_path",
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
    parser.add_argument(
        "--backend",
        choices=["function", "notebook"],
        default="function",
        help="Run independent Python functions by default, or execute notebooks.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Input image path. Mapped to the selected notebook's input variable.",
    )
    parser.add_argument(
        "--mask",
        default=None,
        help="Mask image path for functions that use a mask, such as inpainting.",
    )
    parser.add_argument(
        "--factor",
        type=int,
        default=None,
        help="Super-resolution/downsampling factor for notebooks that define factor.",
    )
    parser.add_argument(
        "--num-iter",
        type=int,
        default=None,
        help="Override num_iter in the selected notebook.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override LR in the selected notebook.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device for function backend: cuda, mps, cpu, or auto when omitted.",
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Override a notebook variable. VALUE is parsed as a Python literal when possible.",
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
        print("%-24s %-32s %s" % (name, FUNCTION_NOTEBOOKS[name], FUNCTION_SCRIPTS[name]))


def parse_param_value(value):
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def parse_param_assignment(assignment):
    if "=" not in assignment:
        raise ValueError("--param must use NAME=VALUE syntax: %s" % assignment)
    name, value = assignment.split("=", 1)
    name = name.strip()
    if not name.isidentifier():
        raise ValueError("--param name must be a valid Python identifier: %s" % name)
    return name, parse_param_value(value.strip())


def collect_overrides(function_name, args):
    overrides = {}

    if args.input is not None:
        input_param = INPUT_PARAM_BY_FUNCTION.get(function_name)
        if input_param is None:
            raise ValueError("--input is not supported for function '%s'" % function_name)
        overrides[input_param] = args.input

    if args.mask is not None:
        mask_param = MASK_PARAM_BY_FUNCTION.get(function_name)
        if mask_param is None:
            raise ValueError("--mask is not supported for function '%s'" % function_name)
        overrides[mask_param] = args.mask

    if args.factor is not None:
        overrides["factor"] = args.factor

    if args.num_iter is not None:
        overrides["num_iter"] = args.num_iter

    if args.lr is not None:
        overrides["LR"] = args.lr

    for assignment in args.param:
        name, value = parse_param_assignment(assignment)
        overrides[name] = value

    return overrides


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


def make_setup_cell(run_dir, function_name, overrides):
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
DIP_CLI_PARAMS = {repr(overrides)}
"""
    return nbformat.v4.new_code_cell(source.strip() + "\n")


def apply_parameter_overrides(nb, overrides):
    if not overrides:
        return {}

    replacements = {name: 0 for name in overrides}
    assignment_patterns = {
        name: re.compile(r"^(\s*)%s\s*=.*$" % re.escape(name))
        for name in overrides
    }

    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue

        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)

        new_lines = []
        for line in source.splitlines(keepends=True):
            newline = "\n" if line.endswith("\n") else ""
            raw_line = line[:-1] if newline else line
            replacement = None

            for name, pattern in assignment_patterns.items():
                match = pattern.match(raw_line)
                if match and not raw_line.lstrip().startswith("#"):
                    replacement = "%s%s = %r%s" % (
                        match.group(1),
                        name,
                        overrides[name],
                        newline,
                    )
                    replacements[name] += 1
                    break

            new_lines.append(replacement if replacement is not None else line)

        cell["source"] = "".join(new_lines)

    missing = {name: value for name, value in overrides.items() if replacements[name] == 0}
    if missing:
        source = "\n".join("%s = %r" % (name, value) for name, value in missing.items())
        nb.cells.insert(1, make_override_cell(source))

    return replacements


def make_override_cell(source):
    import nbformat

    return nbformat.v4.new_code_cell("# Injected CLI parameter overrides.\n" + source + "\n")


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


def build_function_command(function_name, args, timestamp):
    script = REPO_ROOT / FUNCTION_SCRIPTS[function_name]
    if not script.exists():
        raise FileNotFoundError("Function script not found: %s" % script)

    output_dir = build_run_dir(args.output_root, timestamp, function_name)
    cmd = [sys.executable, str(script), "--output-dir", str(output_dir)]

    if args.input is not None:
        if function_name == "flash_no_flash":
            cmd += ["--flash", args.input]
        else:
            cmd += ["--input", args.input]

    if args.mask is not None:
        if function_name != "inpainting":
            raise ValueError("--mask is not supported for function '%s'" % function_name)
        cmd += ["--mask", args.mask]

    if args.factor is not None and function_name in ["super_resolution", "sr_prior_effect"]:
        cmd += ["--factor", str(args.factor)]
    elif args.factor is not None:
        raise ValueError("--factor is not supported for function '%s'" % function_name)

    if args.num_iter is not None:
        cmd += ["--num-iter", str(args.num_iter)]

    if args.lr is not None:
        cmd += ["--lr", str(args.lr)]

    if args.device is not None:
        cmd += ["--device", args.device]

    if args.param:
        raise ValueError("--param is only supported with --backend notebook")

    return cmd, output_dir


def execute_function(function_name, args, timestamp):
    cmd, output_dir = build_function_command(function_name, args, timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"
    run_info_path = output_dir / "run.json"

    info = {
        "function": function_name,
        "backend": "function",
        "command": cmd,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "status": "running",
        "output_dir": display_path(output_dir),
        "stdout": display_path(stdout_path),
        "stderr": display_path(stderr_path),
    }
    write_run_info(run_info_path, info)

    with stdout_path.open("w") as stdout_file, stderr_path.open("w") as stderr_file:
        completed = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=stdout_file, stderr=stderr_file)

    info.update(
        {
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "returncode": completed.returncode,
            "status": "success" if completed.returncode == 0 else "failed",
        }
    )
    write_run_info(run_info_path, info)
    return completed.returncode == 0, info


def execute_notebook(function_name, notebook_name, args, timestamp):
    overrides = collect_overrides(function_name, args)

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
        "parameters": overrides,
    }
    write_run_info(run_info_path, info)

    nb = nbformat.read(notebook_path, as_version=4)
    nb.cells.insert(0, make_setup_cell(run_dir, function_name, overrides))
    replacements = apply_parameter_overrides(nb, overrides)
    info["parameter_replacements"] = replacements
    write_run_info(run_info_path, info)

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
        if args.backend == "notebook":
            try:
                collect_overrides(name, args)
            except ValueError as exc:
                print("error: %s" % exc, file=sys.stderr)
                return 2

            print("Running %s (%s)" % (name, notebook))
            try:
                ok, info = execute_notebook(name, notebook, args, timestamp)
            except ValueError as exc:
                print("error: %s" % exc, file=sys.stderr)
                return 2
        else:
            print("Running %s (%s)" % (name, FUNCTION_SCRIPTS[name]))
            try:
                ok, info = execute_function(name, args, timestamp)
            except ValueError as exc:
                print("error: %s" % exc, file=sys.stderr)
                return 2

        all_ok = all_ok and ok
        print("%s: %s -> %s" % (name, info["status"], info["output_dir"]))
        if not ok and not args.allow_errors:
            break

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
