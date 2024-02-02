import torch


def maybe_to_gpu(data):
    device = get_available_device()

    if isinstance(data, list):
        return [d.to(device, non_blocking=True) for d in data]
    return data.to(device, non_blocking=True)


def get_available_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
