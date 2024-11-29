import numpy as np
import logging
import matplotlib.pyplot as plt


def get_segm_train_fig_with_inp_out_tar(input, output, target, fig_title):
    # This needs to handle the following cases:
    # Segmentation      : {"input": (m,x,y(,z)), "target": (1,x,y(,z)), "output": (c,x,y(,z))}

    channel_idx = np.random.randint(0, input.shape[0])

    if len(input.shape) == 4:
        if len(target[0].nonzero()[0]) > 0:
            foreground_locations = target[0].nonzero()
            slice_to_visualize = foreground_locations[0][np.random.randint(0, len(foreground_locations[0]))]
        else:
            slice_to_visualize = np.random.randint(0, input.shape[1])

        input = input[:, slice_to_visualize]
        if len(target.shape) == 4:
            target = target[:, slice_to_visualize]
        if len(output.shape) == 4:
            output = output[:, slice_to_visualize]

    image = input[channel_idx]
    target = target[0]
    output = output.argmax(0)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), dpi=100, constrained_layout=True)
    axes[0].imshow(image, cmap="gray", vmin=np.quantile(image, 0.01), vmax=np.quantile(image, 0.99))
    axes[0].set_title("input")
    axes[1].imshow(target, cmap="gray")
    axes[1].set_title("target")
    axes[2].imshow(output, cmap="gray")
    axes[2].set_title("output")
    fig.suptitle(fig_title, fontsize=16)
    return fig


def get_cls_train_fig_with_inp_out_tar(input, output, target, fig_title):
    # This needs to handle the following case:
    # Classification    : {"input": (m,x,y(,z)), "target": (n_classes), "output": (n_classes)}

    channel_idx = np.random.randint(0, input.shape[0])

    slice_to_visualize = np.random.randint(0, input.shape[1])

    if len(input.shape) == 4:  # 3D images.
        input = input[:, slice_to_visualize]

    image = input[channel_idx]

    output = output.argmax(0)

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5, 5), dpi=100, constrained_layout=True)
    axes.imshow(image, cmap="gray", vmin=np.quantile(image, 0.01), vmax=np.quantile(image, 0.99))
    axes.set_title(f"Input: {fig_title}", fontsize=12)
    fig.suptitle(f"Target: {target} | Output: {output}", fontsize=12)
    return fig


def get_ssl_train_fig_with_inp_out_tar(input, output, target, fig_title):
    # This needs to handle the following cases:
    # Self-supervised   : {"input": (m,x,y(,z)), "target": (m,x,y(,z)), "output": (m,x,y(,z))}

    channel_idx = np.random.randint(0, input.shape[0])

    if len(input.shape) == 4:  # 3D images.
        slice_to_visualize = np.random.randint(0, input.shape[1])
        input = input[:, slice_to_visualize]
        if len(target.shape) == 4:
            target = target[:, slice_to_visualize]
        if len(output.shape) == 4:
            output = output[:, slice_to_visualize]

    image = input[channel_idx]

    target = target[channel_idx]
    output = output[channel_idx]

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), dpi=100, constrained_layout=True)
    axes[0].imshow(image, cmap="gray", vmin=np.quantile(image, 0.01), vmax=np.quantile(image, 0.99))
    axes[0].set_title("input")
    axes[1].imshow(target, cmap="gray")
    axes[1].set_title("target")
    axes[2].imshow(output, cmap="gray")
    axes[2].set_title("output")
    fig.suptitle(fig_title, fontsize=16)
    return fig
