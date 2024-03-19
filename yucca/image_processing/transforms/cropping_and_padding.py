import numpy as np
from yucca.image_processing.transforms.YuccaTransform import YuccaTransform
from typing import Literal, Union


class CropPad(YuccaTransform):
    def __init__(
        self,
        data_key="image",
        label_key="label",
        pad_value: Union[Literal["min", "zero", "edge"], int, float] = "min",
        patch_size: tuple | list = None,
        p_oversample_foreground=0.0,
    ):
        self.data_key = data_key
        self.label_key = label_key
        self.pad_value = pad_value
        self.patch_size = patch_size
        self.p_oversample_foreground = p_oversample_foreground

    @staticmethod
    def get_params(data, pad_value, target_shape):
        if pad_value == "min":
            pad_kwargs = {"constant_values": data.min(), "mode": "constant"}
        elif pad_value == "zero":
            pad_kwargs = {"constant_values": np.zeros(1, dtype=data.dtype), "mode": "constant"}
        elif isinstance(pad_value, int) or isinstance(pad_value, float):
            pad_kwargs = {"constant_values": pad_value, "mode": "constant"}
        elif pad_value == "edge":
            pad_kwargs = {"mode": "edge"}
        else:
            print("Unrecognized pad value detected.")
        input_shape = data.shape
        target_image_shape = (input_shape[0], *target_shape)
        target_label_shape = (1, *target_shape)
        return input_shape, target_image_shape, target_label_shape, pad_kwargs

    def __croppad__(
        self,
        data_dict: np.ndarray,
        image_properties: dict,
        input_shape: np.ndarray,
        target_image_shape: list | tuple,
        target_label_shape: list | tuple,
        **pad_kwargs,
    ):
        image = data_dict[self.data_key]
        if data_dict.get(self.label_key) is not None:
            label = data_dict[self.label_key]
        else:
            label = None

        if len(self.patch_size) == 3:
            image, label = self.generate_3D_case_from_3D(
                image, image_properties, label, target_image_shape, target_label_shape, **pad_kwargs
            )
        elif len(self.patch_size) == 2 and len(input_shape) == 4:
            image, label = self.generate_2D_case_from_3D(
                image, image_properties, label, target_image_shape, target_label_shape, **pad_kwargs
            )
        elif len(self.patch_size) == 2 and len(input_shape) == 3:
            image, label = self.generate_2D_case_from_2D(
                image, image_properties, label, target_image_shape, target_label_shape, **pad_kwargs
            )
        data_dict[self.data_key] = image
        if label is not None:
            data_dict[self.label_key] = label
        return data_dict

    def generate_3D_case_from_3D(self, image, image_properties, label, target_image_shape, target_label_shape, **pad_kwargs):
        image_out = np.zeros(target_image_shape)
        label_out = np.zeros(target_label_shape)

        # First we pad to ensure min size is met
        to_pad = []
        for d in range(3):
            if image.shape[d + 1] < self.patch_size[d]:
                to_pad += [(self.patch_size[d] - image.shape[d + 1])]
            else:
                to_pad += [0]

        pad_lb_x = to_pad[0] // 2
        pad_ub_x = to_pad[0] // 2 + to_pad[0] % 2
        pad_lb_y = to_pad[1] // 2
        pad_ub_y = to_pad[1] // 2 + to_pad[1] % 2
        pad_lb_z = to_pad[2] // 2
        pad_ub_z = to_pad[2] // 2 + to_pad[2] % 2

        # This is where we should implement any patch selection biases.
        # The final patch excted after augmentation will always be the center of this patch
        # to avoid interpolation artifacts near the borders
        crop_start_idx = []
        if len(image_properties["foreground_locations"]) == 0 or np.random.uniform() >= self.p_oversample_foreground:
            for d in range(3):
                if image.shape[d + 1] < self.patch_size[d]:
                    crop_start_idx += [0]
                else:
                    crop_start_idx += [np.random.randint(image.shape[d + 1] - self.patch_size[d] + 1)]
        else:
            locidx = np.random.choice(len(image_properties["foreground_locations"]))
            location = image_properties["foreground_locations"][locidx]
            for d in range(3):
                if image.shape[d + 1] < self.patch_size[d]:
                    crop_start_idx += [0]
                else:
                    crop_start_idx += [
                        np.random.randint(
                            max(0, location[d] - self.patch_size[d]),
                            min(location[d], image.shape[d + 1] - self.patch_size[d]) + 1,
                        )
                    ]

        image_out[
            :,
            :,
            :,
            :,
        ] = np.pad(
            image[
                :,
                crop_start_idx[0] : crop_start_idx[0] + self.patch_size[0],
                crop_start_idx[1] : crop_start_idx[1] + self.patch_size[1],
                crop_start_idx[2] : crop_start_idx[2] + self.patch_size[2],
            ],
            ((0, 0), (pad_lb_x, pad_ub_x), (pad_lb_y, pad_ub_y), (pad_lb_z, pad_ub_z)),
            **pad_kwargs,
        )
        if label is None:
            return image_out, None
        label_out[
            :,
            :,
            :,
            :,
        ] = np.pad(
            label[
                :,
                crop_start_idx[0] : crop_start_idx[0] + self.patch_size[0],
                crop_start_idx[1] : crop_start_idx[1] + self.patch_size[1],
                crop_start_idx[2] : crop_start_idx[2] + self.patch_size[2],
            ],
            ((0, 0), (pad_lb_x, pad_ub_x), (pad_lb_y, pad_ub_y), (pad_lb_z, pad_ub_z)),
        )
        return image_out, label_out

    def generate_2D_case_from_3D(self, image, image_properties, label, target_image_shape, target_label_shape, **pad_kwargs):
        """
        The possible input for this can be 2D or 3D data.
        For 2D we want to pad or crop as necessary.
        For 3D we want to first select a slice from the first dimension, i.e. volume[idx, :, :],
        then pad or crop as necessary.
        """
        image_out = np.zeros(target_image_shape)
        label_out = np.zeros(target_label_shape)

        # First we pad to ensure min size is met
        to_pad = []
        for d in range(2):
            if image.shape[d + 2] < self.patch_size[d]:
                to_pad += [(self.patch_size[d] - image.shape[d + 2])]
            else:
                to_pad += [0]

        pad_lb_y = to_pad[0] // 2
        pad_ub_y = to_pad[0] // 2 + to_pad[0] % 2
        pad_lb_z = to_pad[1] // 2
        pad_ub_z = to_pad[1] // 2 + to_pad[1] % 2

        # This is where we should implement any patch selection biases.
        # The final patch extracted after augmentation will always be the center of this patch
        # as this is where augmentation-induced interpolation artefacts are least likely
        crop_start_idx = []
        if len(image_properties["foreground_locations"]) == 0 or np.random.uniform() >= self.p_oversample_foreground:
            x_idx = np.random.randint(image.shape[1])
            for d in range(2):
                if image.shape[d + 2] < self.patch_size[d]:
                    crop_start_idx += [0]
                else:
                    crop_start_idx += [np.random.randint(image.shape[d + 2] - self.patch_size[d] + 1)]
        else:
            locidx = np.random.choice(len(image_properties["foreground_locations"]))
            location = image_properties["foreground_locations"][locidx]
            x_idx = location[0]
            for d in range(2):
                if image.shape[d + 2] < self.patch_size[d]:
                    crop_start_idx += [0]
                else:
                    crop_start_idx += [
                        np.random.randint(
                            max(0, location[d + 1] - self.patch_size[d]),
                            min(location[d + 1], image.shape[d + 2] - self.patch_size[d]) + 1,
                        )
                    ]

        image_out[:, :, :] = np.pad(
            image[
                :,
                x_idx,
                crop_start_idx[0] : crop_start_idx[0] + self.patch_size[0],
                crop_start_idx[1] : crop_start_idx[1] + self.patch_size[1],
            ],
            ((0, 0), (pad_lb_y, pad_ub_y), (pad_lb_z, pad_ub_z)),
            **pad_kwargs,
        )

        if label is None:
            return image_out, None

        label_out[:, :, :] = np.pad(
            label[
                :,
                x_idx,
                crop_start_idx[0] : crop_start_idx[0] + self.patch_size[0],
                crop_start_idx[1] : crop_start_idx[1] + self.patch_size[1],
            ],
            ((0, 0), (pad_lb_y, pad_ub_y), (pad_lb_z, pad_ub_z)),
        )

        return image_out, label_out

    def generate_2D_case_from_2D(self, image, image_properties, label, target_image_shape, target_label_shape, **pad_kwargs):
        """
        The possible input for this can be 2D or 3D data.
        For 2D we want to pad or crop as necessary.
        For 3D we want to first select a slice from the first dimension, i.e. volume[idx, :, :],
        then pad or crop as necessary.
        """
        image_out = np.zeros(target_image_shape)
        label_out = np.zeros(target_label_shape)

        # First we pad to ensure min size is met
        to_pad = []
        for d in range(2):
            if image.shape[d + 1] < self.patch_size[d]:
                to_pad += [(self.patch_size[d] - image.shape[d + 1])]
            else:
                to_pad += [0]

        pad_lb_x = to_pad[0] // 2
        pad_ub_x = to_pad[0] // 2 + to_pad[0] % 2
        pad_lb_y = to_pad[1] // 2
        pad_ub_y = to_pad[1] // 2 + to_pad[1] % 2

        # This is where we should implement any patch selection biases.
        # The final patch extracted after augmentation will always be the center of this patch
        # as this is where artefacts are least present
        crop_start_idx = []
        if len(image_properties["foreground_locations"]) == 0 or np.random.uniform() >= self.p_oversample_foreground:
            for d in range(2):
                if image.shape[d + 1] < self.patch_size[d]:
                    crop_start_idx += [0]
                else:
                    crop_start_idx += [np.random.randint(image.shape[d + 1] - self.patch_size[d] + 1)]
        else:
            locidx = np.random.choice(len(image_properties["foreground_locations"]))
            location = image_properties["foreground_locations"][locidx]
            for d in range(2):
                if image.shape[d + 1] < self.patch_size[d]:
                    crop_start_idx += [0]
                else:
                    crop_start_idx += [
                        np.random.randint(
                            max(0, location[d] - self.patch_size[d]),
                            min(location[d], image.shape[d + 1] - self.patch_size[d]) + 1,
                        )
                    ]

        image_out[:, :, :] = np.pad(
            image[
                :,
                crop_start_idx[0] : crop_start_idx[0] + self.patch_size[0],
                crop_start_idx[1] : crop_start_idx[1] + self.patch_size[1],
            ],
            ((0, 0), (pad_lb_x, pad_ub_x), (pad_lb_y, pad_ub_y)),
            **pad_kwargs,
        )

        if label is None:  # Reconstruction/inpainting
            return image_out, None

        if len(label.shape) == 1:  # Classification
            return image_out, label

        label_out[:, :, :] = np.pad(
            label[
                :,
                crop_start_idx[0] : crop_start_idx[0] + self.patch_size[0],
                crop_start_idx[1] : crop_start_idx[1] + self.patch_size[1],
            ],
            ((0, 0), (pad_lb_x, pad_ub_x), (pad_lb_y, pad_ub_y)),
        )

        return image_out, label_out

    def __call__(self, packed_data_dict=None, image_properties=None, **unpacked_data_dict):
        data_dict = packed_data_dict if packed_data_dict else unpacked_data_dict

        input_shape, target_image_shape, target_label_shape, pad_kwargs = self.get_params(
            data=data_dict[self.data_key], pad_value=self.pad_value, target_shape=self.patch_size
        )

        data_dict = self.__croppad__(
            data_dict=data_dict,
            image_properties=image_properties,
            input_shape=input_shape,
            target_image_shape=target_image_shape,
            target_label_shape=target_label_shape,
            **pad_kwargs,
        )
        return data_dict
