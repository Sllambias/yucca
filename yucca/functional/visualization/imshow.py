import numpy as np
import logging
import matplotlib.pyplot as plt


def get_train_fig_with_inp_out_tar(input, output, target, fig_title, task_type: str = "segmentation"):
    # This needs to handle the following cases:
    # Segmentation      : {"input": (m,x,y(,z)), "target": (1,x,y(,z)), "output": (c,x,y(,z))}
    # Self-supervised   : {"input": (m,x,y(,z)), "target": (m,x,y(,z)), "output": (m,x,y(,z))}
    # Classification    : {"input": (m,x,y(,z)), "target": (1,x), "output": (c,x)}

    channel_idx = np.random.randint(0, input.shape[0])

    if len(input.shape) == 4:  # 3D images.
        # We need to select a slice to visualize.
        if task_type == "segmentation" and len(target[0].nonzero()[0]) > 0:
            # Select a foreground slice if any exist.
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

    if task_type in ["segmentation", "classification"]:
        target = target[0]
        output = output.argmax(0)
    elif task_type == "self-supervised":
        target = target[channel_idx]
        output = output[channel_idx]
    else:
        logging.warn(
            f"Unknown task type. Found {task_type} and expected one in ['classification', 'segmentation', 'self-supervised']"
        )

    if len(target.shape) == 1:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5, 5), dpi=100, constrained_layout=True)
        axes[0].imshow(image, cmap="gray", vmin=np.quantile(image, 0.01), vmax=np.quantile(image, 0.99))
        axes[0].set_title("input")
        fig.suptitle(f"{fig_title}. Target: {target} | Output: {output}", fontsize=16)
    else:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), dpi=100, constrained_layout=True)
        axes[0].imshow(image, cmap="gray", vmin=np.quantile(image, 0.01), vmax=np.quantile(image, 0.99))
        axes[0].set_title("input")
        axes[1].imshow(target, cmap="gray")
        axes[1].set_title("target")
        axes[2].imshow(output, cmap="gray")
        axes[2].set_title("output")
        fig.suptitle(fig_title, fontsize=16)
    return fig
