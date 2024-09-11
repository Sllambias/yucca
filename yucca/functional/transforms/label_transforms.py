import numpy as np


def translate_region_labels(regions: dict, labels: dict) -> dict:
    str_label_to_int = {v: int(k) for k, v in labels.items()}

    regions_with_int_labels = {}
    for region, region_dict in regions.items():
        regions_with_int_labels[region] = {
            "priority": region_dict["priority"],
            "labels": [str_label_to_int[label] for label in region_dict["labels"]],
        }
    return regions_with_int_labels


def convert_labels_to_regions(label, regions):
    b, c, *shape = label.shape
    assert c == 1, f"Class labels are not onehot encoded. Channel dim must be 1, but got c={c}."
    region_canvas = np.zeros((b, len(regions), *shape))

    for channel, region in enumerate(regions.keys()):
        region_labels = regions[region]["labels"]
        region_canvas[:, channel] = np.isin(label[:, 0], region_labels)
    region_canvas = region_canvas.astype(np.uint8)

    return region_canvas


# Currently not used, and incorrect with new regions format.

# def convert_regions_to_labels(data, region_labels):
#     b, _, *shape = data.shape
#     region_canvas = np.zeros((b, 1, *shape))
#     for channel, region_label in enumerate(region_labels):
#         mask = data[:, channel] > 0.5
#         region_canvas[:, 0][mask] = region_label
#     region_canvas = region_canvas
#     return region_canvas


# def torch_convert_regions_to_labels(data, region_labels):
#     b, _, *shape = data.shape
#     region_canvas = torch.zeros((b, 1, *shape))
#     for channel, region_label in enumerate(region_labels):
#         mask = data[:, channel] > 0.5
#         region_canvas[:, 0][mask] = region_label
#     region_canvas = region_canvas
#     return region_canvas
