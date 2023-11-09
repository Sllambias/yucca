import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_pickle
from yucca.training.data_loading.YuccaLoader import YuccaLoader


class YuccaLoader_Classification(YuccaLoader):
    def determine_shapes(self):
        # Here we only load the image and not the seg.
        # The pairs will be of shape (2,) since it's an object array
        image, _ = self.load_and_maybe_keep_volume(self.files[0])

        self.input_shape = image.shape
        self.image_shape_for_aug = (self.batch_size, image.shape[0], *self.gen_patch_size)
        self.seg_shape_for_aug = (self.batch_size, 1)
        del image

    def generate_3D_batch(self):
        """
        The possible input for this can be 2D or 3D data.
        For 2D we want to pad or crop as necessary.
        For 3D we want to first select a slice from the first dimension, i.e. volume[idx, :, :],
        then pad or crop as necessary.
        """
        image_batch = np.zeros(self.image_shape_for_aug)
        seg_batch = np.zeros(self.seg_shape_for_aug)

        samples = np.random.choice(self.files, self.batch_size)

        for idx, sample in enumerate(samples):
            image, seg = self.load_and_maybe_keep_volume(sample)
            image_properties = self.load_and_maybe_keep_pickle(sample[:-4] + ".pkl")

            assert len(image.shape) == 4, "input should be (c, x, y, z)" f" but it is: {image.shape}"

            # First we pad to ensure min size is met
            to_pad = []
            for d in range(3):
                if image.shape[d + 1] < self.gen_patch_size[d]:
                    to_pad += [(self.gen_patch_size[d] - image.shape[d + 1])]
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
            # as this is where artefacts are least present
            crop_start_idx = []
            if len(image_properties["foreground_locations"]) == 0 or np.random.uniform() >= self.p_oversample_foreground:
                for d in range(3):
                    if image.shape[d + 1] < self.gen_patch_size[d]:
                        crop_start_idx += [0]
                    else:
                        crop_start_idx += [np.random.randint(image.shape[d + 1] - self.gen_patch_size[d] + 1)]
            else:
                locidx = np.random.choice(len(image_properties["foreground_locations"]))
                location = image_properties["foreground_locations"][locidx]
                for d in range(3):
                    if image.shape[d + 1] < self.gen_patch_size[d]:
                        crop_start_idx += [0]
                    else:
                        crop_start_idx += [
                            np.random.randint(
                                max(0, location[d] - self.gen_patch_size[d]),
                                min(location[d], image.shape[d + 1] - self.gen_patch_size[d]) + 1,
                            )
                        ]

            image_batch[
                idx,
                :,
                :,
                :,
                :,
            ] = np.pad(
                image[
                    :,
                    crop_start_idx[0] : crop_start_idx[0] + self.gen_patch_size[0],
                    crop_start_idx[1] : crop_start_idx[1] + self.gen_patch_size[1],
                    crop_start_idx[2] : crop_start_idx[2] + self.gen_patch_size[2],
                ],
                ((0, 0), (pad_lb_x, pad_ub_x), (pad_lb_y, pad_ub_y), (pad_lb_z, pad_ub_z)),
                mode="edge",
            )

            seg_batch[idx, :] = seg

        data_dict = {"image": image_batch, "seg": seg_batch}
        return data_dict

    def generate_2D_batch_from_3D(self):
        """
        The possible input for this can be 2D or 3D data.
        For 2D we want to pad or crop as necessary.
        For 3D we want to first select a slice from the first dimension, i.e. volume[idx, :, :],
        then pad or crop as necessary.
        """
        image_batch = np.zeros(self.image_shape_for_aug)
        seg_batch = np.zeros(self.seg_shape_for_aug)

        samples = np.random.choice(self.files, self.batch_size)

        for idx, sample in enumerate(samples):
            image, seg = self.load_and_maybe_keep_volume(sample)
            image_properties = self.load_and_maybe_keep_pickle(sample[:-4] + ".pkl")

            assert len(image.shape) == 4, "input should be (c, x, y, z)" f" but it is: {image.shape}"

            # First we pad to ensure min size is met
            to_pad = []
            for d in range(2):
                if image.shape[d + 2] < self.gen_patch_size[d]:
                    to_pad += [(self.gen_patch_size[d] - image.shape[d + 2])]
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
                    if image.shape[d + 2] < self.gen_patch_size[d]:
                        crop_start_idx += [0]
                    else:
                        crop_start_idx += [np.random.randint(image.shape[d + 2] - self.gen_patch_size[d] + 1)]
            else:
                locidx = np.random.choice(len(image_properties["foreground_locations"]))
                location = image_properties["foreground_locations"][locidx]
                x_idx = location[0]
                for d in range(2):
                    if image.shape[d + 2] < self.gen_patch_size[d]:
                        crop_start_idx += [0]
                    else:
                        crop_start_idx += [
                            np.random.randint(
                                max(0, location[d + 1] - self.gen_patch_size[d]),
                                min(location[d + 1], image.shape[d + 2] - self.gen_patch_size[d]) + 1,
                            )
                        ]

            image_batch[idx, :, :, :] = np.pad(
                image[
                    :,
                    x_idx,
                    crop_start_idx[0] : crop_start_idx[0] + self.gen_patch_size[0],
                    crop_start_idx[1] : crop_start_idx[1] + self.gen_patch_size[1],
                ],
                ((0, 0), (pad_lb_y, pad_ub_y), (pad_lb_z, pad_ub_z)),
                mode="edge",
            )

            seg_batch[idx, :] = seg

        data_dict = {"image": image_batch, "seg": seg_batch}
        return data_dict

    def generate_2D_batch_from_2D(self):
        """
        The possible input for this can be 2D or 3D data.
        For 2D we want to pad or crop as necessary.
        For 3D we want to first select a slice from the first dimension, i.e. volume[idx, :, :],
        then pad or crop as necessary.
        """
        image_batch = np.zeros(self.image_shape_for_aug)
        seg_batch = np.zeros(self.seg_shape_for_aug)

        samples = np.random.choice(self.files, self.batch_size)

        for idx, sample in enumerate(samples):
            image, seg = self.load_and_maybe_keep_volume(sample)
            image_properties = self.load_and_maybe_keep_pickle(sample[:-4] + ".pkl")

            assert len(image.shape) == 3, "input should be (c, x, y)" f" but it is: {image.shape}"

            # First we pad to ensure min size is met
            to_pad = []
            for d in range(2):
                if image.shape[d + 1] < self.gen_patch_size[d]:
                    to_pad += [(self.gen_patch_size[d] - image.shape[d + 1])]
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
                    if image.shape[d + 1] < self.gen_patch_size[d]:
                        crop_start_idx += [0]
                    else:
                        crop_start_idx += [np.random.randint(image.shape[d + 1] - self.gen_patch_size[d] + 1)]
            else:
                locidx = np.random.choice(len(image_properties["foreground_locations"]))
                location = image_properties["foreground_locations"][locidx]
                for d in range(2):
                    if image.shape[d + 1] < self.gen_patch_size[d]:
                        crop_start_idx += [0]
                    else:
                        crop_start_idx += [
                            np.random.randint(
                                max(0, location[d] - self.gen_patch_size[d]),
                                min(location[d], image.shape[d + 1] - self.gen_patch_size[d]) + 1,
                            )
                        ]

            image_batch[idx, :, :, :] = np.pad(
                image[
                    :,
                    crop_start_idx[0] : crop_start_idx[0] + self.gen_patch_size[0],
                    crop_start_idx[1] : crop_start_idx[1] + self.gen_patch_size[1],
                ],
                ((0, 0), (pad_lb_x, pad_ub_x), (pad_lb_y, pad_ub_y)),
                mode="edge",
            )

            seg_batch[idx, :] = seg

        data_dict = {"image": image_batch, "seg": seg_batch}
        return data_dict
