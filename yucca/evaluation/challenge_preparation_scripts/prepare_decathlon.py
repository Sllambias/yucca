from batchgenerators.utilities.file_and_folder_operations import subfiles, join, maybe_mkdir_p
import shutil

"""
Task031_
Task032_
Task033_
Task034_
Task035_
Task036_
Task037_
Task038_
Task039_
Task040_
"""

prefixes = [
    "DC_BrainTumour_",
    "DC_Heart_",
    "DC_Liver_",
    "DC_Hippocampus_",
    "DC_Prostate_",
    "DC_Lung_",
    "DC_Pancreas_",
    "DC_HepaticVessel_",
    "DC_Spleen_",
    "DC_Colon_",
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
    "/home/zcr545/YuccaData/yucca_predictions/Task031_DC_BrainTumour/Task031_DC_BrainTumour/UNet3D/YuccaTrainerV2__YuccaPlannerV2/fold_0_checkpoint_best",
    "/home/zcr545/YuccaData/yucca_predictions/Task032_DC_Heart/Task032_DC_Heart/UNet3D/YuccaTrainerV2__YuccaPlannerV2/fold_0_checkpoint_best",
    "/home/zcr545/YuccaData/yucca_predictions/Task033_DC_Liver/Task033_DC_Liver/UNet3D/YuccaTrainerV2__YuccaPlannerV2/fold_0_checkpoint_best",
    "/home/zcr545/YuccaData/yucca_predictions/Task034_DC_Hippocampus/Task034_DC_Hippocampus/UNet3D/YuccaTrainerV2__YuccaPlannerV2/fold_0_checkpoint_best",
    "/home/zcr545/YuccaData/yucca_predictions/Task035_DC_Prostate/Task035_DC_Prostate/UNet3D/YuccaTrainerV2__YuccaPlannerV2/fold_0_checkpoint_best",
    "/home/zcr545/YuccaData/yucca_predictions/Task036_DC_Lung/Task036_DC_Lung/UNet3D/YuccaTrainerV2__YuccaPlannerV2/fold_0_checkpoint_best",
    "/home/zcr545/YuccaData/yucca_predictions/Task037_DC_Pancreas/Task037_DC_Pancreas/UNet3D/YuccaTrainerV2__YuccaPlannerV2/fold_0_checkpoint_best",
    "/home/zcr545/YuccaData/yucca_predictions/Task038_DC_HepaticVessel/Task038_DC_HepaticVessel/UNet3D/YuccaTrainerV2__YuccaPlannerV2/fold_0_checkpoint_best",
    "/home/zcr545/YuccaData/yucca_predictions/Task039_DC_Spleen/Task039_DC_Spleen/UNet3D/YuccaTrainerV2__YuccaPlannerV2/fold_0_checkpoint_best",
    "/home/zcr545/YuccaData/yucca_predictions/Task040_DC_Colon/Task040_DC_Colon/UNet3D/YuccaTrainerV2__YuccaPlannerV2/fold_0_checkpoint_best",
]

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

savepath = "/home/zcr545/decathlon_submission"
maybe_mkdir_p(savepath)

for i in range(10):
    for seg_path in subfiles(folders[i], join=False):
        new_filename = seg_path[len(prefixes[i]) :]
        new_path = join(expected[i], new_filename)
        if new_path in blacklist:
            print("blocking: ", new_path)
            continue
        maybe_mkdir_p(join(savepath, expected[i]))

        shutil.copy(join(folders[i], seg_path), f"{savepath}/{expected[i]}/{new_filename}")
