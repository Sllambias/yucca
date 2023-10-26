#%%
from batchgenerators.utilities.file_and_folder_operations import *
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

#%%

datasets = {
        'MSSEG-1':
            {'low': {"Prediction:": "/home/zcr545/YuccaData/yucca_segmentations/Task054_MSSeg/Task052_3BrainLesionFTBalanced/UNet2D/YuccaTrainerV2_FT__YuccaPlannerV2Z/fold_0_checkpoint_best/Center_07MSSeg_Patient_10.nii.gz",
                     "Ground Truth:": "/home/zcr545/YuccaData/yucca_raw_data/Task054_MSSeg/labelsTs/Center_07MSSeg_Patient_10.nii.gz",
                     "Base Image:":"/home/zcr545/YuccaData/yucca_raw_data/Task054_MSSeg/imagesTs/Center_07MSSeg_Patient_10_000.nii.gz"},
             'high': {"Prediction:": "/home/zcr545/YuccaData/yucca_segmentations/Task054_MSSeg/Task052_3BrainLesionFTBalanced/UNet2D/YuccaTrainerV2_FT__YuccaPlannerV2Z/fold_0_checkpoint_best/Center_03MSSeg_Patient_07.nii.gz",
                     "Ground Truth:": "/home/zcr545/YuccaData/yucca_raw_data/Task054_MSSeg/labelsTs/Center_03MSSeg_Patient_07.nii.gz",
                     "Base Image:":"/home/zcr545/YuccaData/yucca_raw_data/Task054_MSSeg/imagesTs/Center_03MSSeg_Patient_07_000.nii.gz"}},
        'MS08':
            {'low': {"Prediction:": "/home/zcr545/YuccaData/yucca_segmentations/Task041_MSLesion08/Task052_3BrainLesionFTBalanced/UNet2D/YuccaTrainerV2_FT__YuccaPlannerV2Z/fold_0_checkpoint_best/MSLesion_UNC_train_Case07.nii.gz",
                     "Ground Truth:": "/home/zcr545/YuccaData/yucca_raw_data/Task041_MSLesion08/labelsTs/MSLesion_UNC_train_Case07.nii.gz",
                     "Base Image:":"/home/zcr545/YuccaData/yucca_raw_data/Task041_MSLesion08/imagesTs/MSLesion_UNC_train_Case07_000.nii.gz"},
             'high': {"Prediction:": "/home/zcr545/YuccaData/yucca_segmentations/Task041_MSLesion08/Task052_3BrainLesionFTBalanced/UNet2D/YuccaTrainerV2_FT__YuccaPlannerV2Z/fold_0_checkpoint_best/MSLesion_CHB_train_Case08.nii.gz",
                      "Ground Truth:": "/home/zcr545/YuccaData/yucca_raw_data/Task041_MSLesion08/labelsTs/MSLesion_CHB_train_Case08.nii.gz",    
                     "Base Image:":"/home/zcr545/YuccaData/yucca_raw_data/Task041_MSLesion08/imagesTs/MSLesion_CHB_train_Case08_000.nii.gz"}},
        'Stroke':
            {'low': {"Prediction:": "/home/zcr545/YuccaData/yucca_segmentations/Task042_ISLES22/Task052_3BrainLesionFTBalanced/UNet2D/YuccaTrainerV2_FT__YuccaPlannerV2Z/fold_0_checkpoint_best/ISLES22_sub-strokecase0020.nii.gz",
                     "Ground Truth:": "/home/zcr545/YuccaData/yucca_raw_data/Task042_ISLES22/labelsTs/ISLES22_sub-strokecase0020.nii.gz",
                     "Base Image:": "/home/zcr545/YuccaData/yucca_raw_data/Task042_ISLES22/imagesTs/ISLES22_sub-strokecase0020_000.nii.gz"},
             'high': {"Prediction:": "/home/zcr545/YuccaData/yucca_segmentations/Task042_ISLES22/Task052_3BrainLesionFTBalanced/UNet2D/YuccaTrainerV2_FT__YuccaPlannerV2Z/fold_0_checkpoint_best/ISLES22_sub-strokecase0055.nii.gz",
                      "Ground Truth:": "/home/zcr545/YuccaData/yucca_raw_data/Task042_ISLES22/labelsTs/ISLES22_sub-strokecase0055.nii.gz",
                      "Base Image:":"/home/zcr545/YuccaData/yucca_raw_data/Task042_ISLES22/imagesTs/ISLES22_sub-strokecase0055_000.nii.gz"}},
        'WMH':
            {'low': {"Prediction:": "/home/zcr545/YuccaData/yucca_segmentations/Task003_WMH_Flair/Task052_3BrainLesionFTBalanced/UNet2D/YuccaTrainerV2_FT__YuccaPlannerV2Z/fold_0_checkpoint_best/WMH_F_154.nii.gz",
                     "Ground Truth:": "/home/zcr545/YuccaData/yucca_raw_data/Task003_WMH_Flair/labelsTs/WMH_F_154.nii.gz",
                     "Base Image:": "/home/zcr545/YuccaData/yucca_raw_data/Task003_WMH_Flair/imagesTs/WMH_F_154_000.nii.gz"},
            'high': {"Prediction:": "/home/zcr545/YuccaData/yucca_segmentations/Task003_WMH_Flair/Task052_3BrainLesionFTBalanced/UNet2D/YuccaTrainerV2_FT__YuccaPlannerV2Z/fold_0_checkpoint_best/WMH_F_151.nii.gz",
                     "Ground Truth:": "/home/zcr545/YuccaData/yucca_raw_data/Task003_WMH_Flair/labelsTs/WMH_F_151.nii.gz",
                     "Base Image:": "/home/zcr545/YuccaData/yucca_raw_data/Task003_WMH_Flair/imagesTs/WMH_F_151_000.nii.gz"}}
        }


plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
#plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 16


cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["green", "red", "orange"])

fig, ax = plt.subplots(2, 4, figsize=(8,4), subplot_kw=dict(box_aspect=1))

for x, dataset in enumerate(datasets):
    if dataset in ['MSSEG-1', 'MS08']:
        rots = 3
    else:
        rots = 1
    for y, load in enumerate(datasets[dataset]):
        if datasets[dataset][load] == None:
            print(dataset)
            continue
        pred = nib.load(datasets[dataset][load]['Prediction:']).get_fdata()
        gt = nib.load(datasets[dataset][load]['Ground Truth:']).get_fdata()
        base_image = nib.load(datasets[dataset][load]['Base Image:']).get_fdata()

        sliceidx = np.argsort(gt.sum((0,1)))[-1]

#[gt.shape[2]//4:-gt.shape[2]//4]
        base_slice = np.rot90(base_image[:,:,sliceidx], rots)
        nonzero = np.nonzero(base_slice)
        ymin = min(nonzero[0])
        ymax = max(nonzero[0])
        xmin = min(nonzero[1]) 
        xmax = max(nonzero[1]) 

        if not dataset == 'MS08':
            xmin = max(0, xmin-20)
            xmax = min(base_slice.shape[-1], xmax+20)
        if dataset == 'MS08':
            ymin += 40
            xmin += 40
            ymax -= 50
            xmax -= 50
        base_slice = base_slice[ymin:ymax, xmin:xmax]
        gt_slice = gt[:,:,sliceidx] + (pred[:,:,sliceidx]*2)
        gt_slice = np.rot90(np.ma.masked_where(gt_slice == 0, gt_slice), rots)
        gt_slice = gt_slice[ymin:ymax, xmin:xmax]
        ax[y, x].imshow(base_slice, cmap='gray', aspect='auto')
        ax[y, x].imshow(gt_slice, cmap=cmap1, alpha=1, aspect='auto')
        ax[y, x].set_xticks([])
        ax[y, x].set_yticks([])
        #plt.imshow(pred_slice, cmap=cmap2, alpha=0.5)
for axe, name in zip(ax[0], list(datasets.keys())): 
    axe.set_title(name)
for axe, name in zip(ax[:,0], ["Low", "High"]): 
    axe.set_ylabel(name)

plt.subplots_adjust(wspace=0, hspace=0)
fig.savefig('/home/zcr545/seg_examples.pdf', bbox_inches='tight', dpi=1400)

plt.show()        
#%%
