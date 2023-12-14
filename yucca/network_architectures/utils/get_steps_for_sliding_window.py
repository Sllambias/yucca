import numpy as np


def get_steps_for_sliding_window(shape: np.array, patch_size: tuple, overlap: int):
    assert overlap >= 0 and overlap < 1, "make sure overlap is 0 =< overlap < 1"
    steplist = []

    for i, patch in enumerate(patch_size):
        # To compute the steps for an image with dimensions=(251, 214, 198)
        # and patch size=(64, 128, 96):
        # First get all the possible steps between 0 and 251
        # with step size=64 * 1-overlap.
        # i.e. overlap = 0 means we step a full patch each time,
        slices = slice(0, shape[i], np.floor(patch * (1 - overlap)).astype(int))
        steps = np.arange(shape[i])[slices]

        # Then we get the final step start-idx and remove any steps starting
        # at a higher idx than this.
        # For a dimension of 251 pixels/voxels with a patch size of 64
        # the last start-idx will be 251-64=187.
        final_step = shape[i] - patch
        steps = steps[steps < final_step]
        steps = np.append(steps, final_step)

        steplist.append(steps)
    return steplist
