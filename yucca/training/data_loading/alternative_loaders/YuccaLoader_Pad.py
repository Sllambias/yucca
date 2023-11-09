import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_pickle
from time import localtime, strftime, time, mktime
from yucca.training.data_loading.YuccaLoader import YuccaLoader


class YuccaLoader_Pad(YuccaLoader):
    """
    Based on the YuccaLoader but includes padding in the preprocessing
    - This lets us control the padding mode, e.g. 'edge' and 'constant'
    """

    def __init__(
        self, list_of_files, batch_size, gen_patch_size, p_oversample_foreground=1.0, random_state=None, keep_in_ram=True
    ):
        super().__init__(list_of_files, batch_size, gen_patch_size, p_oversample_foreground, random_state, keep_in_ram)

    def generate_3D_batch(self):
        image_batch = np.zeros(self.image_shape_for_aug)
        seg_batch = np.zeros(self.seg_shape_for_aug)

        samples = np.random.choice(self.files, self.batch_size)

        for idx, sample in enumerate(samples):
            imseg = self.load_path(sample)
            # First we pad to ensure min size is met
            to_pad = []
            for d in range(3):
                if imseg.shape[d + 1] < self.gen_patch_size[d]:
                    to_pad += [(self.gen_patch_size[d] - imseg.shape[d + 1])]
                else:
                    to_pad += [0]

            pad_lb_x = to_pad[0] // 2
            pad_ub_x = to_pad[0] // 2 + to_pad[0] % 2
            pad_lb_y = to_pad[1] // 2
            pad_ub_y = to_pad[1] // 2 + to_pad[1] % 2
            pad_lb_z = to_pad[2] // 2
            pad_ub_z = to_pad[2] // 2 + to_pad[2] % 2
            offsets = [pad_lb_x, pad_lb_y, pad_lb_z]

            image = np.pad(imseg[:-1], ((0, 0), (pad_lb_x, pad_ub_x), (pad_lb_y, pad_ub_y), (pad_lb_z, pad_ub_z)), mode="edge")
            seg = np.pad(
                imseg[-1:], ((0, 0), (pad_lb_x, pad_ub_x), (pad_lb_y, pad_ub_y), (pad_lb_z, pad_ub_z)), mode="constant"
            )

            assert image.shape[1:] == seg.shape[1:], "image and seg shapes are not identical"

            # When minimums are ensured by padding we crop to initial patch size.
            # This is where we should implement any patch selection biases.
            # The final patch extracted after augmentation will always be the center of this patch
            # as this is where artefacts are least present
            crop_start_idx = []
            if np.random.uniform() < self.p_oversample_foreground:
                image_properties = self.load_and_maybe_keep_pickle(sample[:-4] + ".pkl")
                locidx = np.random.choice(len(image_properties["foreground_locations"]))
                location = image_properties["foreground_locations"][locidx]
                for d in range(3):
                    crop_start_idx += [
                        np.random.randint(
                            max(0, location[d] + offsets[d] - self.gen_patch_size[d]),
                            min(location[d] + offsets[d], image.shape[d + 1] - self.gen_patch_size[d]) + 1,
                        )
                    ]
            else:
                for d in range(3):
                    crop_start_idx += [np.random.randint(image.shape[d + 1] - self.gen_patch_size[d] + 1)]
            image_batch[idx] = image[
                :,
                crop_start_idx[0] : crop_start_idx[0] + self.gen_patch_size[0],
                crop_start_idx[1] : crop_start_idx[1] + self.gen_patch_size[1],
                crop_start_idx[2] : crop_start_idx[2] + self.gen_patch_size[2],
            ]
            seg_batch[idx] = seg[
                :,
                crop_start_idx[0] : crop_start_idx[0] + self.gen_patch_size[0],
                crop_start_idx[1] : crop_start_idx[1] + self.gen_patch_size[1],
                crop_start_idx[2] : crop_start_idx[2] + self.gen_patch_size[2],
            ]

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
            imseg = self.load_path(sample)

            assert len(imseg.shape) == 4, "input should be (c, x, y, z)" f" but it is: {imseg.shape}"

            # First we pad to ensure min size is met
            to_pad = []
            for d in range(2):
                if imseg.shape[d + 2] < self.gen_patch_size[d]:
                    to_pad += [(self.gen_patch_size[d] - imseg.shape[d + 2])]
                else:
                    to_pad += [0]
            pad_lb_y = to_pad[0] // 2
            pad_ub_y = to_pad[0] // 2 + to_pad[0] % 2
            pad_lb_z = to_pad[1] // 2
            pad_ub_z = to_pad[1] // 2 + to_pad[1] % 2
            offsets = [pad_lb_y, pad_lb_z]

            image = np.pad(imseg[:-1], ((0, 0), (0, 0), (pad_lb_y, pad_ub_y), (pad_lb_z, pad_ub_z)), mode="edge")
            seg = np.pad(imseg[-1:], ((0, 0), (0, 0), (pad_lb_y, pad_ub_y), (pad_lb_z, pad_ub_z)), mode="constant")

            assert image.shape[1:] == seg.shape[1:], "image and seg shapes are not identical"

            # When minimums are ensured by padding we crop to initial patch size.
            # This is where we should implement any patch selection biases.
            # The final patch extracted after augmentation will always be the center of this patch
            # as this is where artefacts are least present
            crop_start_idx = []
            if np.random.uniform() < self.p_oversample_foreground:
                image_properties = self.load_and_maybe_keep_pickle(sample[:-4] + ".pkl")
                locidx = np.random.choice(len(image_properties["foreground_locations"]))
                location = image_properties["foreground_locations"][locidx]
                x_idx = location[0]
                for d in range(2):
                    crop_start_idx += [
                        np.random.randint(
                            max(0, location[d + 1] + offsets[d] - self.gen_patch_size[d]),
                            min(location[d + 1] + offsets[d], image.shape[d + 2] - self.gen_patch_size[d]) + 1,
                        )
                    ]
            else:
                x_idx = np.random.randint(image.shape[1])
                for d in range(2):
                    crop_start_idx += [np.random.randint(image.shape[d + 2] - self.gen_patch_size[d] + 1)]
            image_batch[idx] = image[
                :,
                x_idx,
                crop_start_idx[0] : crop_start_idx[0] + self.gen_patch_size[0],
                crop_start_idx[1] : crop_start_idx[1] + self.gen_patch_size[1],
            ]
            seg_batch[idx] = seg[
                :,
                x_idx,
                crop_start_idx[0] : crop_start_idx[0] + self.gen_patch_size[0],
                crop_start_idx[1] : crop_start_idx[1] + self.gen_patch_size[1],
            ]

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
            imseg = self.load_path(sample)
            assert len(imseg.shape) == 3, "input should be (c, x, y)" f" but it is: {imseg.shape}"

            # First we pad to ensure min size is met
            to_pad = []
            for d in range(2):
                if imseg.shape[d + 1] < self.gen_patch_size[d]:
                    to_pad += [(self.gen_patch_size[d] - imseg.shape[d + 1])]
                else:
                    to_pad += [0]

            pad_lb_x = to_pad[0] // 2
            pad_ub_x = to_pad[0] // 2 + to_pad[0] % 2
            pad_lb_y = to_pad[1] // 2
            pad_ub_y = to_pad[1] // 2 + to_pad[1] % 2
            offsets = [pad_lb_x, pad_lb_y]

            image = np.pad(imseg[:-1], ((0, 0), (pad_lb_x, pad_ub_x), (pad_lb_y, pad_ub_y)), mode="edge")
            seg = np.pad(imseg[-1:], ((0, 0), (pad_lb_x, pad_ub_x), (pad_lb_y, pad_ub_y)), mode="constant")

            assert image.shape[1:] == seg.shape[1:], "image and seg shapes are not identical"

            # When minimums are ensured by padding we crop to initial patch size.
            # This is where we should implement any patch selection biases.
            # The final patch extracted after augmentation will always be the center of this patch
            # as this is where artefacts are least present
            crop_start_idx = []
            if np.random.uniform() < self.p_oversample_foreground:
                image_properties = self.load_and_maybe_keep_pickle(sample[:-4] + ".pkl")
                locidx = np.random.choice(len(image_properties["foreground_locations"]))
                location = image_properties["foreground_locations"][locidx]
                for d in range(2):
                    crop_start_idx += [
                        np.random.randint(
                            max(0, location[d] + offsets[d] - self.gen_patch_size[d]),
                            min(location[d] + offsets[d], image.shape[d + 1] - self.gen_patch_size[d]) + 1,
                        )
                    ]
            else:
                for d in range(2):
                    crop_start_idx += [np.random.randint(image.shape[d + 1] - self.gen_patch_size[d] + 1)]
            image_batch[idx] = image[
                :,
                crop_start_idx[0] : crop_start_idx[0] + self.gen_patch_size[0],
                crop_start_idx[1] : crop_start_idx[1] + self.gen_patch_size[1],
            ]
            seg_batch[idx] = seg[
                :,
                crop_start_idx[0] : crop_start_idx[0] + self.gen_patch_size[0],
                crop_start_idx[1] : crop_start_idx[1] + self.gen_patch_size[1],
            ]

        data_dict = {"image": image_batch, "seg": seg_batch}
        return data_dict
