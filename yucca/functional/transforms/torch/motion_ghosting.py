import torch


def torch_motion_ghosting(
    image: torch.Tensor,
    alpha: float,
    num_reps: int,
    axis: int,
    clip_to_input_range: bool = False,
) -> torch.Tensor:
    img_min = image.min()
    img_max = image.max()
    m = min(0, img_min.item())
    image = image + abs(m)

    if image.ndim == 3:
        image = torch.fft.fftn(image, dim=[0, 1, 2])

        if axis == 0:
            image[0:-1:num_reps, :, :] = alpha * image[0:-1:num_reps, :, :]
        elif axis == 1:
            image[:, 0:-1:num_reps, :] = alpha * image[:, 0:-1:num_reps, :]
        else:
            image[:, :, 0:-1:num_reps] = alpha * image[:, :, 0:-1:num_reps]

        image = torch.abs(torch.fft.ifftn(image, dim=[0, 1, 2]))

    elif image.ndim == 2:
        image = torch.fft.fftn(image, dim=[0, 1])

        if axis == 0:
            image[0:-1:num_reps, :] = alpha * image[0:-1:num_reps, :]
        else:
            image[:, 0:-1:num_reps] = alpha * image[:, 0:-1:num_reps]

        image = torch.abs(torch.fft.ifftn(image, dim=[0, 1]))

    image = image - abs(m)

    if clip_to_input_range:
        image = torch.clamp(image, min=img_min, max=img_max)

    return image
