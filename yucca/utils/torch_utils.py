import torch


def maybe_to_cuda(data):
    if not torch.cuda.is_available():
        return data
    if isinstance(data, list):
        return [d.to("cuda", non_blocking=True) for d in data]
    return data.to("cuda", non_blocking=True)
