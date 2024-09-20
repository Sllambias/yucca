from batchgenerators.utilities.file_and_folder_operations import subfiles, join, maybe_mkdir_p as ensure_dir_exists
from SimpleITK import ReadImage
import glob
import os
import logging
import SimpleITK as sitk


"""
Task031_MSD_BrainTumour
Task032_MSD_Heart
Task033_MSD_Liver
Task034_MSD_Hippocampus
Task035_MSD_Prostate
Task036_MSD_Lung
Task037_MSD_Pancreas
Task038_MSD_HepaticVessel
Task039_MSD_Spleen
Task040_MSD_Colon
"""


current = [
    "Task031_MSD_BrainTumour",
    "Task032_MSD_Heart",
    "Task033_MSD_Liver",
    "Task034_MSD_Hippocampus",
    "Task035_MSD_Prostate",
    "Task036_MSD_Lung",
    "Task037_MSD_Pancreas",
    "Task038_MSD_HepaticVessel",
    "Task039_MSD_Spleen",
    "Task040_MSD_Colon",
]

expected = [
    "Task01_BrainTumour",
    "Task02_Heart",
    "Task03_Liver",
    "Task04_Hippocampus",
    "Task05_Prostate",
    "Task06_Lung",
    "Task07_Pancreas",
    "Task08_HepaticVessel",
    "Task09_Spleen",
    "Task10_Colon",
]

folders = [
    "/projects/zcr545_shared_datasets/people/zcr545/YuccaData/yucca_results/Task031_MSD_BrainTumour/Task031_MSD_BrainTumour/UNet__3D/YuccaManagerV11__YuccaPlannerV3/default/kfold_5_fold_4/version_0/best",
    "/projects/zcr545_shared_datasets/people/zcr545/YuccaData/yucca_results/Task032_MSD_Heart/Task032_MSD_Heart/UNet__3D/YuccaManagerV11__YuccaPlannerV3/default/kfold_5_fold_4/version_0/best",
    "/projects/zcr545_shared_datasets/people/zcr545/YuccaData/yucca_results/Task033_MSD_Liver/Task033_MSD_Liver/UNet__3D/YuccaManagerV11__YuccaPlannerV3/default/kfold_5_fold_4/version_0/best",
    "/projects/zcr545_shared_datasets/people/zcr545/YuccaData/yucca_results/Task034_MSD_Hippocampus/Task034_MSD_Hippocampus/UNet__3D/YuccaManagerV11__YuccaPlannerV3/default/kfold_5_fold_4/version_0/best",
    "/projects/zcr545_shared_datasets/people/zcr545/YuccaData/yucca_results/Task035_MSD_Prostate/Task035_MSD_Prostate/UNet__3D/YuccaManagerV11__YuccaPlannerV3/default/kfold_5_fold_3/version_0/best",
    "/projects/zcr545_shared_datasets/people/zcr545/YuccaData/yucca_results/Task036_MSD_Lung/Task036_MSD_Lung/UNet__3D/YuccaManagerV11__YuccaPlannerV3/default/kfold_5_fold_4/version_0/best",
    "/projects/zcr545_shared_datasets/people/zcr545/YuccaData/yucca_results/Task037_MSD_Pancreas/Task037_MSD_Pancreas/UNet__3D/YuccaManagerV11__YuccaPlannerV3/default/kfold_5_fold_4/version_0/best",
    "/projects/zcr545_shared_datasets/people/zcr545/YuccaData/yucca_results/Task038_MSD_HepaticVessel/Task038_MSD_HepaticVessel/UNet__3D/YuccaManagerV11__YuccaPlannerV3/default/kfold_5_fold_4/version_0/best",
    "/projects/zcr545_shared_datasets/people/zcr545/YuccaData/yucca_results/Task039_MSD_Spleen/Task039_MSD_Spleen/UNet__3D/YuccaManagerV11__YuccaPlannerV3/default/kfold_5_fold_4/version_0/best",
    "/projects/zcr545_shared_datasets/people/zcr545/YuccaData/yucca_results/Task040_MSD_Colon/Task040_MSD_Colon/UNet__3D/YuccaManagerV11__YuccaPlannerV3/default/kfold_5_fold_4/version_0/best",
]

exclude = [
    "Task03_Liver/liver_141.nii.gz",
    "Task03_Liver/liver_156.nii.gz",
    "Task03_Liver/liver_160.nii.gz",
    "Task03_Liver/liver_161.nii.gz",
    "Task03_Liver/liver_162.nii.gz",
    "Task03_Liver/liver_164.nii.gz",
    "Task03_Liver/liver_167.nii.gz",
    "Task03_Liver/liver_182.nii.gz",
    "Task03_Liver/liver_189.nii.gz",
    "Task03_Liver/liver_190.nii.gz",
    "Task08_HepaticVessel/hepaticvessel_247.nii.gz",
]

savepath = "/home/zcr545/decathlon_submission"
ensure_dir_exists(savepath)

blacklist = [
    "Task03_Liver/liver_141.nii.gz",
    "Task03_Liver/liver_156.nii.gz",
    "Task03_Liver/liver_160.nii.gz",
    "Task03_Liver/liver_161.nii.gz",
    "Task03_Liver/liver_162.nii.gz",
    "Task03_Liver/liver_164.nii.gz",
    "Task03_Liver/liver_167.nii.gz",
    "Task03_Liver/liver_182.nii.gz",
    "Task03_Liver/liver_189.nii.gz",
    "Task03_Liver/liver_190.nii.gz",
    "Task08_HepaticVessel/hepaticvessel_247.nii.gz",
]

for i in range(10):
    for seg_path in subfiles(folders[i], join=False):
        new_filename = seg_path[len(current[i]) :]
        new_path = join(expected[i], new_filename)
        if new_path in blacklist:
            print("blocking: ", new_path)
            continue
        ensure_dir_exists(join(savepath, expected[i]))


def copy_geometry(image: sitk.Image, ref: sitk.Image):
    if ref.GetDimension() == 4:
        ref = ref[:, :, :, 0]
    print(image.GetSize(), ref.GetSize(), ref.GetDimension())
    image.SetOrigin(ref.GetOrigin())
    image.SetDirection(ref.GetDirection())
    image.SetSpacing(ref.GetSpacing())
    return image


logger = logging.getLogger()
sub_dir = "/home/zcr545/decathlon_submission"
gt_dir = "/home/zcr545/data/data/public_datasets/decathlon"


for i in range(10):
    img_list = [x for x in glob.glob(os.path.join(sub_dir, expected[i], "*.nii.gz"))]
    for predpath in img_list:
        pred = ReadImage(predpath)
        name = os.path.split(predpath)[-1]
        gtpath = os.path.join(gt_dir, expected[i], "imagesTs", name)
        gt = ReadImage(gtpath)
        copy_geometry(pred, gt)
# %%
