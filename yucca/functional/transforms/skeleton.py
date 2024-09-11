import numpy as np
from skimage.morphology import skeletonize, dilation


def skeleton(label, do_tube: bool = True):
    # Add tubed skeleton GT
    label_copy = label
    bin_seg = label_copy > 0
    seg_all_skel = np.zeros_like(bin_seg, dtype=np.int16)

    # Skeletonize
    if not np.sum(bin_seg) == 0:
        skel = skeletonize(bin_seg)
        skel = (skel > 0).astype(np.int16)
        if do_tube:
            skel = dilation(dilation(skel))
        skel *= label_copy.astype(np.int16)
    return seg_all_skel
