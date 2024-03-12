import torch
from fvcore.nn import FlopCountAnalysis


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


def flush_and_get_torch_memory_allocated(device):
    if isinstance(device, torch.device):
        device = device.type  # Get the name: str of the torch.device such as "cuda", "mps" or "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()
        return torch.cuda.memory_allocated(device)
    elif device == "mps":
        torch.mps.empty_cache()
        return torch.mps.driver_allocated_memory()
    else:
        return 0


def measure_FLOPs(model: torch.nn.Module, data: torch.Tensor):
    # Returns a FlopCountAnalysis object
    # Use obj.total() for total # FLOPs
    # Use obj.by_module() for # FLOPs pr. module
    return FlopCountAnalysis(model, data)
