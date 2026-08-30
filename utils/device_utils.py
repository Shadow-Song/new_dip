import torch


def get_device(prefer_mps=True):
    """Return the best available torch device for this project."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    mps_backend = getattr(torch.backends, "mps", None)
    if prefer_mps and mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def to_device(value, device=None):
    """Move tensors or modules to the selected device."""
    if device is None:
        device = get_device()
    return value.to(device)
