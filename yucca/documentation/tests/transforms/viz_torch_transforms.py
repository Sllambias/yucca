#! /usr/bin/env python3

# How to run:
# python viz_torch_transforms --datasets-dir /path/to/datasets --output /path/to/output.png
# python viz_torch_transforms --datasets-dir /path/to/datasets --output /path/to/output.png --headless
# Defaults: datasets dir is ~/datasets and output is ./viz_torch_transforms.png

import os
import glob
import nibabel as nib
import matplotlib.pyplot as plt
import argparse
import torch
import numpy as np

from yucca.functional.transforms.torch import (
    torch_bias_field,
    torch_blur,
    torch_gamma,
    torch_motion_ghosting,
    torch_mask,
    torch_additive_noise,
    torch_multiplicative_noise,
    torch_gibbs_ringing,
    torch_simulate_lowres,
    torch_spatial,
)

torch.manual_seed(1234)


def show_high_res(data, title):
    """Show a high-resolution view of the data with slice navigation."""
    print(f"\nOpening high-resolution view for: {title}")
    print(f"Data shape: {data.shape}")

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.canvas.manager.set_window_title(f"High Resolution: {title}")
    current_slice = data.shape[2] // 2

    # Show initial image and colorbar
    im = ax.imshow(data[:, :, current_slice], cmap="gray")
    ax.set_title(f"{title}\nSlice {current_slice} of {data.shape[2]}", pad=20)
    # cbar = plt.colorbar(im, ax=ax, label='Intensity')

    def update_display():
        im.set_data(data[:, :, current_slice])
        ax.set_title(f"{title}\nSlice {current_slice} of {data.shape[2]}", pad=20)
        # Optionally, update colorbar limits if data range changes
        # im.set_clim(vmin=data.min(), vmax=data.max())
        fig.canvas.draw_idle()

    def on_key(event):
        nonlocal current_slice
        print(f"Key pressed: {event.key}")
        if event.key == "up" and current_slice < data.shape[2] - 1:
            current_slice += 1
            print(f"Moving to slice {current_slice}")
            update_display()
        elif event.key == "down" and current_slice > 0:
            current_slice -= 1
            print(f"Moving to slice {current_slice}")
            update_display()

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


def on_click(event):
    """Handle click events on the main figure."""
    print(f"\nClick event detected at: {event.xdata}, {event.ydata}")

    if event.inaxes is None:
        print("Click was outside axes")
        return

    ax = event.inaxes
    print(f"Clicked on axes: {ax}")

    for idx, (title, data) in enumerate(transforms.items()):
        if ax == axes[idx // n_cols, idx % n_cols]:
            print(f"Found matching subplot: {title}")
            show_high_res(data.numpy(), title)
            return

    print("No matching subplot found")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply and visualize image transformations")
    parser.add_argument("--headless", action="store_true", help="Run without displaying plots")
    parser.add_argument(
        "--datasets-dir",
        type=str,
        default=os.path.expanduser("~/datasets"),
        help="Path to the datasets directory (default: ~/datasets)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="viz_torch_transforms.png",
        help="Output image file path (default: viz_torch_transforms.png)",
    )
    args = parser.parse_args()

    nii_files = glob.glob(os.path.join(args.datasets_dir, "**", "*.nii.gz"), recursive=True)
    nii_files.sort()

    single_sample = nii_files[1]
    im = nib.load(single_sample)
    data = im.get_fdata()

    print("\nImage Analysis:")
    print("Data type:", data.dtype)
    print("Shape:", data.shape)
    print("Value range:", data.min(), "to", data.max())
    print("Mean value:", data.mean())
    print("Std value:", data.std())
    print("Unique values count:", len(np.unique(data)))
    print("First few unique values:", np.unique(data)[:10])

    # Convert to float32 and scale to [0,1] while preserving relative intensities
    imarr = data.astype(np.float32)
    data_min = imarr.min()
    data_max = imarr.max()
    data_range = data_max - data_min

    # Scale to [0,1] while preserving relative intensities
    imarr = (imarr - data_min) / data_range
    imarr = torch.from_numpy(imarr)
    # Assume batch and channel dimensions are present
    # imarr = imarr.unsqueeze(0).unsqueeze(0)

    print("\nNormalized Image Analysis:")
    print("Value range:", imarr.min().item(), "to", imarr.max().item())
    print("Mean value:", imarr.mean().item())
    print("Std value:", imarr.std().item())

    transforms = {
        "Original": imarr,
        "Bias Field": torch_bias_field(imarr.clone(), clip_to_input_range=True),
        "Blurred (σ=2.0)": torch_blur(imarr.clone(), sigma=2.0),
        "Gamma (0.5–2.0)": torch_gamma(imarr.clone(), gamma_range=(0.5, 2.0), clip_to_input_range=True),
        "Gamma (2.0)": torch_gamma(imarr, gamma_range=(2.0, 2.0), clip_to_input_range=True),
        "Ghost (α=2, ax=0)": torch_motion_ghosting(imarr.clone(), alpha=2.0, num_reps=4, axis=0, clip_to_input_range=True),
        "Ghost (α=2, ax=1)": torch_motion_ghosting(imarr.clone(), alpha=2.0, num_reps=4, axis=1, clip_to_input_range=True),
        "Masked (r=0.3)": torch_mask(imarr.clone(), pixel_value=0, ratio=0.3, token_size=[16, 16, 16]),
        "Add Noise (σ=0.05)": torch_additive_noise(imarr.clone(), mean=0, sigma=0.05, clip_to_input_range=True),
        "Add Noise (σ=0.1)": torch_additive_noise(imarr.clone(), mean=0, sigma=0.1, clip_to_input_range=True),
        "Add Noise (σ=0.2)": torch_additive_noise(imarr.clone(), mean=0, sigma=0.2, clip_to_input_range=True),
        "Mult Noise (σ=0.05)": torch_multiplicative_noise(imarr.clone(), mean=0, sigma=0.05, clip_to_input_range=True),
        "Mult Noise (σ=0.1)": torch_multiplicative_noise(imarr.clone(), mean=0, sigma=0.1, clip_to_input_range=True),
        "Mult Noise (σ=0.2)": torch_multiplicative_noise(imarr.clone(), mean=0, sigma=0.2, clip_to_input_range=True),
        "Gibbs (axes=0,1,2)": torch_gibbs_ringing(
            imarr.clone(), num_sample=64, axes=[0, 1, 2], mode="rect", clip_to_input_range=True
        ),
        "Low Res": torch_simulate_lowres(imarr.clone(), target_shape=(32, 32, 32), clip_to_input_range=True),
        "Spatial": torch_spatial(
            imarr.clone(),
            patch_size=imarr.shape,
            p_deform=1.0,
            p_rot=1.0,
            p_rot_per_axis=1.0,
            p_scale=1.0,
            alpha=10.0,
            sigma=3.0,
            x_rot=0.5,
            y_rot=0,
            z_rot=0,
            scale_factor=1.0,
            clip_to_input_range=True,
        )[
            0
        ],  # Get only the image, not the label
    }

    # Get middle slice for visualization
    slice_idx = imarr.shape[-1] // 2  # Changed to use last dimension

    # Calculate grid dimensions
    n_transforms = len(transforms)
    n_cols = 6  # Show 6 images per row
    n_rows = (n_transforms + n_cols - 1) // n_cols  # Ceiling division

    # Create a figure with subplots
    fig_width = 2.2 * n_cols
    fig_height = 2.2 * n_rows
    plt.rcParams["figure.figsize"] = [fig_width, fig_height]
    plt.rcParams["figure.dpi"] = 200  # Keep high DPI for quality
    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False)

    # Plot each transformation
    for idx, (title, transformed_data) in enumerate(transforms.items()):
        row = idx // n_cols
        col = idx % n_cols

        # Plot transformed image
        if isinstance(transformed_data, tuple):
            transformed_data = transformed_data[0]
        print(title, transformed_data.shape)
        transformed_plot = transformed_data.numpy()
        axes[row, col].imshow(transformed_plot[:, :, slice_idx], cmap="gray")
        axes[row, col].set_title(title, pad=1, fontsize=7)
        axes[row, col].axis("off")

    # Hide any unused subplots
    for idx in range(len(transforms), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis("off")
        axes[row, col].set_visible(False)

    # Use subplots_adjust for tight packing
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.03, right=0.97, wspace=0.02, hspace=0.15)

    # Set window title
    fig.canvas.manager.set_window_title("Image Transformations")

    # Save the figure
    plt.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"\nSaved visualization to: {args.output}")

    # Connect the click event
    print("\nConnecting click event handler...")
    fig.canvas.mpl_connect("button_press_event", on_click)
    print("Click event handler connected. Click any image to view in high resolution.")

    # Only show the plot if not in headless mode
    if not args.headless:
        plt.show()
