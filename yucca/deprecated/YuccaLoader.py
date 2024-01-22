import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_pickle


class YuccaLoader(object):
    """
    Default data loader for patch-based dataloading and augmentation.

    This class takes input data in 2D or 3D, batch size, generator patch size and
    the probability to oversample foreground. If the data is 3D it outputs either 2D or 3D data,
    depending on the model. If the data is 2D it outputs 2D data.

    3D data will be of the shape (b, c, x, y, z) and 2D data will be shape (b, c, x, y)
    with b = batch size, c = modalities (derived from the modalities of the input data) and
    xy(z) defined by the generator patch size.

    The generator patch size is larger than the final patch size to remove border interpolation
    artifacts. If we cropped to the final patch size here, and then performed transformations such
    as rotations or scaling we may get large border artifacts or black areas where there could
    have been parts of the real image, if we started out with a larger area then did the
    augmentations and THEN cropped to the desired size.

    This class also supports set_thread_id and __next__ functions required for the CPU threading and
    background implementation used to apply augmentations in the framework.

    Finally, by default it will keep cases in RAM to save the time required to repeatedly load the
    image, however this should be avoided for very large datasets.
    """

    def __init__(
        self,
        list_of_files,
        batch_size,
        gen_patch_size,
        p_oversample_foreground=0.5,
        keep_in_ram=True,
    ):
        self.batch_size = batch_size
        self.gen_patch_size = gen_patch_size
        self.files = list_of_files
        self.p_oversample_foreground = p_oversample_foreground
        self.keep_in_ram = keep_in_ram
        self.already_loaded_cases = {}
        self.determine_shapes()

    def determine_shapes(self):
        image = self.load_and_maybe_keep_volume(self.files[0])
        self.input_shape = image.shape
        self.image_shape_for_aug = (
            self.batch_size,
            image.shape[0] - 1,
            *self.gen_patch_size,
        )
        self.seg_shape_for_aug = (self.batch_size, 1, *self.gen_patch_size)
        del image

    def generate_train_batch(self):
        if len(self.gen_patch_size) == 3:
            return self.generate_3D_batch()
        if len(self.gen_patch_size) == 2 and len(self.input_shape) == 4:
            return self.generate_2D_batch_from_3D()
        if len(self.gen_patch_size) == 2 and len(self.input_shape) == 3:
            return self.generate_2D_batch_from_2D()
        else:
            print(f"patch size should be (x, y, z) or (x,y) but is: {self.gen_patch_size}")

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
            imseg = self.load_and_maybe_keep_volume(sample)
            image_properties = self.load_and_maybe_keep_pickle(sample[:-4] + ".pkl")

            assert len(imseg.shape) == 4, "input should be (c, x, y, z)" f" but it is: {imseg.shape}"

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

            # This is where we should implement any patch selection biases.
            # The final patch excted after augmentation will always be the center of this patch
            # as this is where artefacts are least present
            crop_start_idx = []
            if len(image_properties["foreground_locations"]) == 0 or np.random.uniform() >= self.p_oversample_foreground:
                for d in range(3):
                    if imseg.shape[d + 1] < self.gen_patch_size[d]:
                        crop_start_idx += [0]
                    else:
                        crop_start_idx += [np.random.randint(imseg.shape[d + 1] - self.gen_patch_size[d] + 1)]
            else:
                locidx = np.random.choice(len(image_properties["foreground_locations"]))
                location = image_properties["foreground_locations"][locidx]
                for d in range(3):
                    if imseg.shape[d + 1] < self.gen_patch_size[d]:
                        crop_start_idx += [0]
                    else:
                        crop_start_idx += [
                            np.random.randint(
                                max(0, location[d] - self.gen_patch_size[d]),
                                min(
                                    location[d],
                                    imseg.shape[d + 1] - self.gen_patch_size[d],
                                )
                                + 1,
                            )
                        ]

            image_batch[
                idx,
                :,
                :,
                :,
                :,
            ] = np.pad(
                imseg[
                    :-1,
                    crop_start_idx[0] : crop_start_idx[0] + self.gen_patch_size[0],
                    crop_start_idx[1] : crop_start_idx[1] + self.gen_patch_size[1],
                    crop_start_idx[2] : crop_start_idx[2] + self.gen_patch_size[2],
                ],
                (
                    (0, 0),
                    (pad_lb_x, pad_ub_x),
                    (pad_lb_y, pad_ub_y),
                    (pad_lb_z, pad_ub_z),
                ),
                mode="edge",
            )

            seg_batch[
                idx,
                :,
                :,
                :,
                :,
            ] = np.pad(
                imseg[
                    -1:,
                    crop_start_idx[0] : crop_start_idx[0] + self.gen_patch_size[0],
                    crop_start_idx[1] : crop_start_idx[1] + self.gen_patch_size[1],
                    crop_start_idx[2] : crop_start_idx[2] + self.gen_patch_size[2],
                ],
                (
                    (0, 0),
                    (pad_lb_x, pad_ub_x),
                    (pad_lb_y, pad_ub_y),
                    (pad_lb_z, pad_ub_z),
                ),
            )

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
            imseg = self.load_and_maybe_keep_volume(sample)
            image_properties = self.load_and_maybe_keep_pickle(sample[:-4] + ".pkl")

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

            # This is where we should implement any patch selection biases.
            # The final patch extracted after augmentation will always be the center of this patch
            # as this is where augmentation-induced interpolation artefacts are least likely
            crop_start_idx = []
            if len(image_properties["foreground_locations"]) == 0 or np.random.uniform() >= self.p_oversample_foreground:
                x_idx = np.random.randint(imseg.shape[1])
                for d in range(2):
                    if imseg.shape[d + 2] < self.gen_patch_size[d]:
                        crop_start_idx += [0]
                    else:
                        crop_start_idx += [np.random.randint(imseg.shape[d + 2] - self.gen_patch_size[d] + 1)]
            else:
                locidx = np.random.choice(len(image_properties["foreground_locations"]))
                location = image_properties["foreground_locations"][locidx]
                x_idx = location[0]
                for d in range(2):
                    if imseg.shape[d + 2] < self.gen_patch_size[d]:
                        crop_start_idx += [0]
                    else:
                        crop_start_idx += [
                            np.random.randint(
                                max(0, location[d + 1] - self.gen_patch_size[d]),
                                min(
                                    location[d + 1],
                                    imseg.shape[d + 2] - self.gen_patch_size[d],
                                )
                                + 1,
                            )
                        ]

            image_batch[idx, :, :, :] = np.pad(
                imseg[
                    :-1,
                    x_idx,
                    crop_start_idx[0] : crop_start_idx[0] + self.gen_patch_size[0],
                    crop_start_idx[1] : crop_start_idx[1] + self.gen_patch_size[1],
                ],
                ((0, 0), (pad_lb_y, pad_ub_y), (pad_lb_z, pad_ub_z)),
                mode="edge",
            )
            seg_batch[idx, :, :, :] = np.pad(
                imseg[
                    -1:,
                    x_idx,
                    crop_start_idx[0] : crop_start_idx[0] + self.gen_patch_size[0],
                    crop_start_idx[1] : crop_start_idx[1] + self.gen_patch_size[1],
                ],
                ((0, 0), (pad_lb_y, pad_ub_y), (pad_lb_z, pad_ub_z)),
            )

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
            imseg = self.load_and_maybe_keep_volume(sample)
            image_properties = self.load_and_maybe_keep_pickle(sample[:-4] + ".pkl")

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

            # This is where we should implement any patch selection biases.
            # The final patch extracted after augmentation will always be the center of this patch
            # as this is where artefacts are least present
            crop_start_idx = []
            if len(image_properties["foreground_locations"]) == 0 or np.random.uniform() >= self.p_oversample_foreground:
                for d in range(2):
                    if imseg.shape[d + 1] < self.gen_patch_size[d]:
                        crop_start_idx += [0]
                    else:
                        crop_start_idx += [np.random.randint(imseg.shape[d + 1] - self.gen_patch_size[d] + 1)]
            else:
                locidx = np.random.choice(len(image_properties["foreground_locations"]))
                location = image_properties["foreground_locations"][locidx]
                for d in range(2):
                    if imseg.shape[d + 1] < self.gen_patch_size[d]:
                        crop_start_idx += [0]
                    else:
                        crop_start_idx += [
                            np.random.randint(
                                max(0, location[d] - self.gen_patch_size[d]),
                                min(
                                    location[d],
                                    imseg.shape[d + 1] - self.gen_patch_size[d],
                                )
                                + 1,
                            )
                        ]

            image_batch[idx, :, :, :] = np.pad(
                imseg[
                    :-1,
                    crop_start_idx[0] : crop_start_idx[0] + self.gen_patch_size[0],
                    crop_start_idx[1] : crop_start_idx[1] + self.gen_patch_size[1],
                ],
                ((0, 0), (pad_lb_x, pad_ub_x), (pad_lb_y, pad_ub_y)),
                mode="edge",
            )

            seg_batch[idx, :, :, :] = np.pad(
                imseg[
                    -1:,
                    crop_start_idx[0] : crop_start_idx[0] + self.gen_patch_size[0],
                    crop_start_idx[1] : crop_start_idx[1] + self.gen_patch_size[1],
                ],
                ((0, 0), (pad_lb_x, pad_ub_x), (pad_lb_y, pad_ub_y)),
            )
        data_dict = {"image": image_batch, "seg": seg_batch}
        return data_dict

    def __next__(self):
        return self.generate_train_batch()

    def set_thread_id(self, thread_id):
        self.thread_id = thread_id

    def load_and_maybe_keep_pickle(self, picklepath):
        if not self.keep_in_ram:
            return load_pickle(picklepath)
        if picklepath in self.already_loaded_cases:
            return self.already_loaded_cases[picklepath]
        self.already_loaded_cases[picklepath] = load_pickle(picklepath)
        return self.already_loaded_cases[picklepath]

    def load_and_maybe_keep_volume(self, path):
        if not self.keep_in_ram:
            if path[-3:] == "npy":
                return np.load(path, "r")
            image = np.load(path)
            assert len(image.files) == 1, (
                "More than one entry in data array. " f"Should only be ['data'] but is {[key for key in image.files]}"
            )
            return image[image.files[0]]

        if path in self.already_loaded_cases:
            return self.already_loaded_cases[path]

        if path[-3:] == "npy":
            try:
                self.already_loaded_cases[path] = np.load(path, "r")
            except ValueError:
                self.already_loaded_cases[path] = np.load(path, allow_pickle=True)
            return self.already_loaded_cases[path]

        image = np.load(path)
        assert len(image.files) == 1, (
            "More than one entry in data array. " f"Should only be ['data'] but is {[key for key in image.files]}"
        )
        self.already_loaded_cases = image[image.files[0]]
        return self.already_loaded_cases[path]
