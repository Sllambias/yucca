# %%
import numpy as np


def convert_labels_to_regions(labelmap, regions):
    
    b, c, *shape = labelmap.shape
    regionmap = 
    for region in regions:
    pass


import nibabel as nib
import matplotlib.pyplot as plt

path = "/Users/zcr545/Desktop/Projects/repos/yucca_data/raw_data/Task018_KITS23/labelsTr/KITS23_case_00002.nii.gz"
seg = nib.load(path).get_fdata()
print(np.unique(seg, return_counts=True))

# %%
np.isin(seg, [2,3])
# %%
