#%%
#############################################
# Get lesion stats

from batchgenerators.utilities.file_and_folder_operations import subfiles, load_pickle, load_json
import numpy as np
folders = {'WMH': '/home/zcr545/YuccaData/yucca_preprocessed/Task003_WMH_Flair/YuccaPlannerV2',
            'MS08' : '/home/zcr545/YuccaData/yucca_preprocessed/Task041_MSLesion08/YuccaPlannerV2',
            'MSSEG' : '/home/zcr545/YuccaData/yucca_preprocessed/Task054_MSSeg/YuccaPlannerV2X',
            'ISLES' : '/home/zcr545/YuccaData/yucca_preprocessed/Task042_ISLES22/YuccaPlannerV2'}
metrics = {'mean_n_cc': 0,
           'std_n_cc': 0,
           'min_n_cc': 0,
           'max_n_cc': 0,
           'empty_scans': 0,
           'mean_vol_cc': 0,
           'std_vol_cc': 0,
           'min_vol_cc': 0,
           'max_vol_cc': 0,
           }

tasks = {'WMH': metrics.copy(),
         'MS08': metrics.copy(),
         'MSSEG': metrics.copy(),
         'ISLES': metrics.copy()}

reference_spacing = [1., 1., 1.]
for task, folder in folders.items():
    n_cc = []
    vol_cc = []
    pkls = subfiles(folder, suffix='.pkl')
    for pkl in pkls:
        pkl_file = load_pickle(pkl)
        current_spacing = pkl_file['new_spacing']
        conversion = np.divide(reference_spacing, current_spacing)

        n_cc.append(pkl_file['n_cc'])
        if isinstance(pkl_file['size_cc'], list):
            for i in pkl_file['size_cc']:
                if np.prod(conversion) * i > 100000:
                    print(pkl)
                vol_cc.append(np.prod(conversion) * i)
        else:
            vol_cc.append(np.prod(conversion) * pkl_file['size_cc'])

    tasks[task]['mean_n_cc'] = int(np.mean(n_cc))
    tasks[task]['std_n_cc'] = int(np.std(n_cc))
    tasks[task]['empty_scans'] = int(n_cc.count(0))
    n_cc = [n for n in n_cc if n > 0]
    tasks[task]['min_n_cc'] = int(min(n_cc))
    tasks[task]['max_n_cc'] = int(max(n_cc))

    tasks[task]['mean_vol_cc'] = int(np.mean(vol_cc))
    tasks[task]['std_vol_cc'] = int(np.std(vol_cc))
    vol_cc = [vol for vol in vol_cc if vol > 0]
    tasks[task]['min_vol_cc'] = int(min(vol_cc)+.01)
    tasks[task]['max_vol_cc'] = int(max(vol_cc))

print(tasks)    
#############################################

#%%

#############################################
# Get OBJ Stats 
from batchgenerators.utilities.file_and_folder_operations import subfiles, load_pickle, load_json
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
from yucca.evaluation.metrics import sensitivity, specificity, precision, f1
import re

folders = {'WMH': None,
            'MS08' : None,
            'MSSEG' : None,
            'COMBINED_BINARY':  '/home/zcr545/YuccaData/yucca_segmentations/Task052_3BrainLesionFTBalanced/Task052_3BrainLesionFTBalanced/UNet2D/YuccaTrainerV2_FT__YuccaPlannerV2Z/fold_0_checkpoint_best/results_OBJ.json',
            'COMBINED_BINARY_AS_MULTI' : '/home/zcr545/YuccaData/yucca_segmentations/Task052_3BrainLesionFTBalanced/Task052_3BrainLesionFTBalanced/UNet2D/YuccaTrainerV2_FT__YuccaPlannerV2Z/fold_0_checkpoint_best/results_OBJ.json',
            'COMBINED_MULTI': '/home/zcr545/YuccaData/yucca_segmentations/Task055_3BrainLesionFT3LabelsBalanced/Task055_3BrainLesionFT3LabelsBalanced/UNet2D/YuccaTrainerV2_FT__YuccaPlannerV2Z/fold_0_checkpoint_best/results_OBJ.json',
            #'COMBINED_MULTI_AS_BINARY' : '/home/zcr545/YuccaData/yucca_segmentations/Task055_3BrainLesionFT3LabelsBalanced/Task055_3BrainLesionFT3LabelsBalanced/UNet2D/YuccaTrainerV2_FT__YuccaPlannerV2Z/fold_0_checkpoint_best/results_OBJ_BINARY.json',
            }
metrics = []
tasks = {'WMH': metrics.copy(),
         'MS08': metrics.copy(),
         'MSSEG': metrics.copy(),
         'ISLES': metrics.copy(),
         'COMBINED_BINARY': metrics.copy(),
         'COMBINED_BINARY_AS_MULTI':metrics.copy(),
         'COMBINED_MULTI': metrics.copy(),
         'COMBINED_MULTI_AS_BINARY': metrics.copy()}


labels_from_mostafa = {

'ms_center1_01_pred.nii.gz':2 ,
'ms_center1_02_pred.nii.gz':1 ,
'ms_center1_03_pred.nii.gz':1 ,
'ms_center1_04_pred.nii.gz':1 ,
'ms_center1_05_pred.nii.gz':1 ,
'ms_center1_06_pred.nii.gz':1 ,
'ms_center1_07_pred.nii.gz':1 ,
'ms_center1_08_pred.nii.gz':1 ,
'ms_center1_09_pred.nii.gz':2 ,
'ms_center1_10_pred.nii.gz':1 ,
'ms_center3_01_pred.nii.gz':1 ,
'ms_center3_02_pred.nii.gz':1 ,
'ms_center3_03_pred.nii.gz':1 ,
'ms_center3_04_pred.nii.gz':1 ,
'ms_center3_05_pred.nii.gz':1 ,
'ms_center3_06_pred.nii.gz':1 ,
'ms_center3_07_pred.nii.gz':1 ,
'ms_center3_08_pred.nii.gz':1 ,
'ms_center7_01_pred.nii.gz':1 ,
'ms_center7_02_pred.nii.gz':2 ,
'ms_center7_03_pred.nii.gz':1 ,
'ms_center7_04_pred.nii.gz':1 ,
'ms_center7_05_pred.nii.gz':1 ,
'ms_center7_06_pred.nii.gz':2 ,
'ms_center7_07_pred.nii.gz':1 ,
'ms_center7_09_pred.nii.gz':1 ,
'ms_center7_10_pred.nii.gz':1 ,
'ms_center8_01_pred.nii.gz':1 ,
'ms_center8_02_pred.nii.gz':1 ,
'ms_center8_03_pred.nii.gz':1 ,
'ms_center8_04_pred.nii.gz':1 ,
'ms_center8_05_pred.nii.gz':1 ,
'ms_center8_06_pred.nii.gz':1 ,
'ms_center8_07_pred.nii.gz':1 ,
'ms_center8_08_pred.nii.gz':3 ,
'ms_center8_09_pred.nii.gz':1 ,
'ms_center8_10_pred.nii.gz':1 ,
'ms_chb_01_pred.nii.gz':1 ,
'ms_chb_08_pred.nii.gz':1 ,
'ms_chb_09_pred.nii.gz':1 ,
'ms_unc_07_pred.nii.gz':2 ,
'ms_unc_09_pred.nii.gz':2 ,
'stroke_0010_pred.nii.gz':2 ,
'stroke_0017_pred.nii.gz':3 ,
'stroke_0019_pred.nii.gz':2 ,
'stroke_0020_pred.nii.gz':2 ,
'stroke_0027_pred.nii.gz':2 ,
'stroke_0028_pred.nii.gz':2 ,
'stroke_0034_pred.nii.gz':2 ,
'stroke_0042_pred.nii.gz':2 ,
'stroke_0043_pred.nii.gz':2 ,
'stroke_0049_pred.nii.gz':2 ,
'stroke_0055_pred.nii.gz':2 ,
'stroke_0056_pred.nii.gz':2 ,
'stroke_0058_pred.nii.gz':2 ,
'stroke_0060_pred.nii.gz':2 ,
'stroke_0061_pred.nii.gz':2 ,
'stroke_0063_pred.nii.gz':2 ,
'stroke_0064_pred.nii.gz':2 ,
'stroke_0066_pred.nii.gz':2 ,
'stroke_0067_pred.nii.gz':2 ,
'stroke_0068_pred.nii.gz':2 ,
'stroke_0076_pred.nii.gz':2 ,
'stroke_0088_pred.nii.gz':2 ,
'stroke_0092_pred.nii.gz':2 ,
'stroke_0093_pred.nii.gz':2 ,
'stroke_0094_pred.nii.gz':2 ,
'stroke_0098_pred.nii.gz':2 ,
'stroke_0101_pred.nii.gz':2 ,
'stroke_0109_pred.nii.gz':2 ,
'stroke_0115_pred.nii.gz':2 ,
'stroke_0120_pred.nii.gz':1 ,
'stroke_0121_pred.nii.gz':2 ,
'stroke_0123_pred.nii.gz':2 ,
'stroke_0126_pred.nii.gz':2 ,
'stroke_0128_pred.nii.gz':1 ,
'stroke_0143_pred.nii.gz':2 ,
'stroke_0158_pred.nii.gz':2 ,
'stroke_0165_pred.nii.gz':2 ,
'stroke_0167_pred.nii.gz':2 ,
'stroke_0169_pred.nii.gz':2 ,
'stroke_0173_pred.nii.gz':2 ,
'stroke_0181_pred.nii.gz':2 ,
'stroke_0186_pred.nii.gz':2 ,
'stroke_0188_pred.nii.gz':2 ,
'stroke_0190_pred.nii.gz':2 ,
'stroke_0195_pred.nii.gz':2 ,
'stroke_0196_pred.nii.gz':2 ,
'stroke_0197_pred.nii.gz':2 ,
'stroke_0201_pred.nii.gz':2 ,
'stroke_0203_pred.nii.gz':2 ,
'stroke_0206_pred.nii.gz':2 ,
'stroke_0207_pred.nii.gz':2 ,
'stroke_0218_pred.nii.gz':2 ,
'stroke_0219_pred.nii.gz':2 ,
'stroke_0224_pred.nii.gz':2 ,
'stroke_0245_pred.nii.gz':2 ,
'stroke_0250_pred.nii.gz':2 ,
'wmh_10_pred.nii.gz':3 ,
'wmh_111_pred.nii.gz':3 ,
'wmh_117_pred.nii.gz':3 ,
'wmh_118_pred.nii.gz':3 ,
'wmh_119_pred.nii.gz':1 ,
'wmh_120_pred.nii.gz':3 ,
'wmh_121_pred.nii.gz':3 ,
'wmh_122_pred.nii.gz':3 ,
'wmh_123_pred.nii.gz':3 ,
'wmh_124_pred.nii.gz':3 ,
'wmh_125_pred.nii.gz':3 ,
'wmh_127_pred.nii.gz':2 ,
'wmh_128_pred.nii.gz':3 ,
'wmh_129_pred.nii.gz':3 ,
'wmh_12_pred.nii.gz':3 ,
'wmh_130_pred.nii.gz':3 ,
'wmh_131_pred.nii.gz':3 ,
'wmh_133_pred.nii.gz':3 ,
'wmh_134_pred.nii.gz':3 ,
'wmh_135_pred.nii.gz':3 ,
'wmh_136_pred.nii.gz':3 ,
'wmh_138_pred.nii.gz':3 ,
'wmh_139_pred.nii.gz':3 ,
'wmh_13_pred.nii.gz':3 ,
'wmh_140_pred.nii.gz':3 ,
'wmh_141_pred.nii.gz':2 ,
'wmh_142_pred.nii.gz':3 ,
'wmh_143_pred.nii.gz':3 ,
'wmh_145_pred.nii.gz':3 ,
'wmh_146_pred.nii.gz':3 ,
'wmh_147_pred.nii.gz':3 ,
'wmh_148_pred.nii.gz':3 ,
'wmh_149_pred.nii.gz':3 ,
'wmh_14_pred.nii.gz':3 ,
'wmh_150_pred.nii.gz':3 ,
'wmh_151_pred.nii.gz':2 ,
'wmh_152_pred.nii.gz':3 ,
'wmh_153_pred.nii.gz':3 ,
'wmh_154_pred.nii.gz':3 ,
'wmh_155_pred.nii.gz':3 ,
'wmh_156_pred.nii.gz':3 ,
'wmh_157_pred.nii.gz':3 ,
'wmh_158_pred.nii.gz':3 ,
'wmh_159_pred.nii.gz':2 ,
'wmh_15_pred.nii.gz':3 ,
'wmh_160_pred.nii.gz':3 ,
'wmh_161_pred.nii.gz':3 ,
'wmh_162_pred.nii.gz':3 ,
'wmh_163_pred.nii.gz':3 ,
'wmh_164_pred.nii.gz':1 ,
'wmh_165_pred.nii.gz':3 ,
'wmh_166_pred.nii.gz':3 ,
'wmh_167_pred.nii.gz':3 ,
'wmh_168_pred.nii.gz':3 ,
'wmh_169_pred.nii.gz':3 ,
'wmh_16_pred.nii.gz':3 ,
'wmh_18_pred.nii.gz':3 ,
'wmh_1_pred.nii.gz':3 ,
'wmh_20_pred.nii.gz':3 ,
'wmh_22_pred.nii.gz':3 ,
'wmh_24_pred.nii.gz':3 ,
'wmh_26_pred.nii.gz':3 ,
'wmh_28_pred.nii.gz':3 ,
'wmh_30_pred.nii.gz':3 ,
'wmh_32_pred.nii.gz':3 ,
'wmh_34_pred.nii.gz':3 ,
'wmh_36_pred.nii.gz':3 ,
'wmh_38_pred.nii.gz':3 ,
'wmh_3_pred.nii.gz':3 ,
'wmh_40_pred.nii.gz':3 ,
'wmh_42_pred.nii.gz':3 ,
'wmh_43_pred.nii.gz':3 ,
'wmh_44_pred.nii.gz':3 ,
'wmh_45_pred.nii.gz':3 ,
'wmh_46_pred.nii.gz':3 ,
'wmh_47_pred.nii.gz':3 ,
'wmh_48_pred.nii.gz':3 ,
'wmh_5_pred.nii.gz':3 ,
'wmh_70_pred.nii.gz':3 ,
'wmh_71_pred.nii.gz':3 ,
'wmh_72_pred.nii.gz':3 ,
'wmh_73_pred.nii.gz':3 ,
'wmh_74_pred.nii.gz':3 ,
'wmh_75_pred.nii.gz':2 ,
'wmh_76_pred.nii.gz':3 ,
'wmh_77_pred.nii.gz':3 ,
'wmh_78_pred.nii.gz':3 ,
'wmh_79_pred.nii.gz':3 ,
'wmh_7_pred.nii.gz':3 ,
'wmh_80_pred.nii.gz':3 ,
'wmh_81_pred.nii.gz':3 ,
'wmh_82_pred.nii.gz':3 ,
'wmh_83_pred.nii.gz':3 ,
'wmh_84_pred.nii.gz':3 ,
'wmh_85_pred.nii.gz':3 ,
'wmh_86_pred.nii.gz':3 ,
'wmh_87_pred.nii.gz':3 ,
'wmh_88_pred.nii.gz':3 ,
'wmh_89_pred.nii.gz':1 ,
'wmh_90_pred.nii.gz':3 ,
'wmh_91_pred.nii.gz':3 ,
'wmh_92_pred.nii.gz':3 ,
'wmh_93_pred.nii.gz':3 ,
'wmh_94_pred.nii.gz':3 ,
'wmh_95_pred.nii.gz':3 ,
'wmh_96_pred.nii.gz':3 ,
'wmh_97_pred.nii.gz':3 ,
'wmh_98_pred.nii.gz':3 ,
'wmh_99_pred.nii.gz':3 ,
'wmh_9_pred.nii.gz':3 ,

}

for dataset, metricfile in folders.items():
    if not metricfile:
        continue
    file = load_json(metricfile)
    predicted_pat = []
    true_pat = []
    MS = []
    Stroke = []
    WMH = []
    violindict = {'MS08': [],
                'MSSEG': [],
                'Stroke': [],
                'WMH - A': [],
                'WMH - S': [],
                'WMH - U': [],
                }
    for subject in file.keys():
        if subject == 'mean':
            continue
        if '2' not in file[subject].keys():
            if "stroke" in subject:
                prefix = 'stroke_'
            elif 'Center' in subject or 'UNC' in subject or 'CHB' in subject:
                prefix = 'ms_'
            else:
                prefix = 'wmh_'
            
            sub_mostafa = subject.replace("3BLFT_", "")
            sub_mostafa = sub_mostafa.replace(".nii.gz", "")
            sub_mostafa = sub_mostafa.replace("train_Case", "")
            sub_mostafa = sub_mostafa.replace("Patient", "")
            sub_mostafa = sub_mostafa.replace("Center_0", "center")
            sub_mostafa = sub_mostafa.replace("sub-strokecase", "")

            #if prefix == 'wmh'
            sub_mostafa = sub_mostafa.lower()
            name_mostafa = prefix+sub_mostafa+"_pred.nii.gz"

            if name_mostafa in labels_from_mostafa.keys():
                pred = labels_from_mostafa[name_mostafa]
            else:
                pred = 0
        else:
            preds = [file[subject]['1']["Total Positives Prediction"],
                     file[subject]['2']["Total Positives Prediction"],
                     file[subject]['3']["Total Positives Prediction"]]
            pred = np.argmax(preds)+1

        if '2' not in file[subject].keys():
            if np.isnan(file[subject]['1']['Dice']):
                continue
        else:
            if np.isnan(file[subject]['1']['Dice']) and np.isnan(file[subject]['2']['Dice']) and np.isnan(file[subject]['3']['Dice']):
                continue
        if 'Center' in subject or 'UNC' in subject or 'CHB' in subject:
            lesion_label = 1

            prefix = 'ms_'
            # Then we have MS

            if '2' not in file[subject].keys():
                label = str(1)
            else: 
                label = str(1)
            if not 'Volume Similarity' in file[subject][label].keys():
                file[subject][label]['Volume Similarity'] = np.nan
            if not '_OBJ sensitivity' in file[subject][label].keys():
                file[subject][label]['_OBJ sensitivity'] = np.nan
            if not '_OBJ precision' in file[subject][label].keys():
                file[subject][label]['_OBJ precision'] = np.nan

            metric_package = [file[subject][label]['_OBJ precision'],
                              file[subject][label]['_OBJ sensitivity'],
                              file[subject][label]['_OBJ F1'],
                              file[subject][label]['Dice'],
                              file[subject][label]['Volume Similarity']]
            if lesion_label != pred and dataset == 'COMBINED_BINARY_AS_MULTI':
                metric_package = [0, 0, 0, 0, 0]
            MS.append(metric_package)
            
            if 'Center' in subject:
                violindict['MSSEG'].append([metric_package[2], metric_package[3]])
            elif 'CHB' in subject or 'UNC' in subject:
                violindict['MS08'].append([metric_package[2], metric_package[3]])
            else:
                print("something is wrong with MS")
                print(subject)
        elif 'stroke' in subject:
            lesion_label = 2

            # Then we have stroke
            prefix = 'stroke_'
            if '2' not in file[subject].keys():
                label = str(1)
            else: 
                label = str(2)
            if not 'Volume Similarity' in file[subject][label].keys():
                file[subject][label]['Volume Similarity'] = np.nan
            if not '_OBJ sensitivity' in file[subject][label].keys():
                file[subject][label]['_OBJ sensitivity'] = np.nan
            if not '_OBJ precision' in file[subject][label].keys():
                file[subject][label]['_OBJ precision'] = np.nan  
            metric_package = [file[subject][label]['_OBJ precision'],
                              file[subject][label]['_OBJ sensitivity'],
                              file[subject][label]['_OBJ F1'],
                              file[subject][label]['Dice'],
                              file[subject][label]['Volume Similarity']]
            if lesion_label != pred and dataset == 'COMBINED_BINARY_AS_MULTI':
                metric_package = [0, 0, 0, 0, 0]
            Stroke.append(metric_package)
            violindict['Stroke'].append([metric_package[2], metric_package[3]])

        else:
            # Else it's WMH
            lesion_label = 3
            prefix = 'wmh_'
            if '2' not in file[subject].keys():
                label = str(1)
            else: 
                label = str(3)
            if not 'Volume Similarity' in file[subject][label].keys():
                file[subject][label]['Volume Similarity'] = np.nan
            if not '_OBJ sensitivity' in file[subject][label].keys():
                file[subject][label]['_OBJ sensitivity'] = np.nan
            if not '_OBJ precision' in file[subject][label].keys():
                file[subject][label]['_OBJ precision'] = np.nan
            metric_package = [file[subject][label]['_OBJ precision'],
                              file[subject][label]['_OBJ sensitivity'],
                              file[subject][label]['_OBJ F1'],
                              file[subject][label]['Dice'],
                              file[subject][label]['Volume Similarity']]
            if lesion_label != pred and dataset == 'COMBINED_BINARY_AS_MULTI':
                metric_package = [0, 0, 0, 0, 0]
            idx = re.split(r'_|\.', subject)[1]
            if int(idx) in range(49):
                violindict['WMH - U'].append([metric_package[2], metric_package[3]])
            elif int(idx) in range(70, 100):
                violindict['WMH - S'].append([metric_package[2], metric_package[3]])
            elif int(idx) in range(111, 170):
                violindict['WMH - A'].append([metric_package[2], metric_package[3]])
            else:
                print("something is wrong with WMH")
                print(subject)
            WMH.append(metric_package)
        
        if '2' not in file[subject].keys():
            o = subject
            subject = subject.replace("3BLFT_", "")
            subject = subject.replace(".nii.gz", "")
            if prefix == "ms_":
                subject = subject.replace("train_Case", "")
                subject = subject.replace("Patient", "")
                subject = subject.replace("Center_0", "center")
            if prefix == 'stroke_':
                subject = subject.replace("sub-strokecase", "")
            
            #if prefix == 'wmh'
            subject = subject.lower()
            name_mostafa = prefix+subject+"_pred.nii.gz"

            if name_mostafa in labels_from_mostafa.keys():
                pred = labels_from_mostafa[name_mostafa]
            else:
                pred = 0
        else:
            preds = [file[subject]['1']["Total Positives Prediction"],
                     file[subject]['2']["Total Positives Prediction"],
                     file[subject]['3']["Total Positives Prediction"]]
            pred = np.argmax(preds)+1
        predicted_pat.append(pred)
        true_pat.append(int(lesion_label))
    
    matrices = multilabel_confusion_matrix(true_pat, predicted_pat, labels=[0,1,2,3])       
    classification_results = []
    for i in range(3):
        tn, fp, fn, tp = matrices[i+1].ravel()
        prec = precision(tp, fp, tn, fn)
        sens = sensitivity(tp, fp, tn, fn)
        F1 = f1(tp, fp, tn, fn)
        classification_results.append([prec, sens, F1])
    print(classification_results)
    #print(matrices)
    #print("")

    tasks[dataset] = {
        #'Violin':violindict,
                      'MSL': np.append(np.round(classification_results[0], 3), list(zip(np.round(np.nanmean(MS, 0), 3), np.round(np.nanstd(MS, 0), 2)))),
                      'STR': np.append(np.round(classification_results[1], 3), list(zip(np.round(np.nanmean(Stroke, 0), 3), np.round(np.nanstd(Stroke, 0), 2)))),
                      'WMH': np.append(np.round(classification_results[2], 3), list(zip(np.round(np.nanmean(WMH, 0), 3), np.round(np.nanstd(WMH, 0), 2)))),
                      'ALL': np.append(np.round(np.mean(classification_results, 0), 3), np.round(np.nanmean(MS+Stroke+WMH, 0), 3))}    

tasks
#############################################

# %%

# Create boxplot
for i in range(2):
    arrs = []
    labels = []
    fig, ax = plt.subplots()
    for n, dset in enumerate(violindict):
        array = np.array(violindict[dset])[:,i]
        arrs.append(array[~np.isnan(array)])
        labels.append(dset)

    plot = ax.boxplot(arrs, notch=True,
                patch_artist=True,  # fill with color
                labels=labels)  # will be used to label x-ticks
    # fill with colors
    colors = ['firebrick', 'firebrick', 'seagreen', 'navy', 'navy', 'navy']
    for patch, color in zip(plot['boxes'], colors):
        patch.set_alpha(0.40)
        patch.set_facecolor(color)
    ax.set_xlabel('Cohort')
    ax.set_ylabel(ylabels[i])

#%%
# Create violinplots

fig1, ax1 = plt.subplots(1, 2, sharey=True)
fig2, ax2 = plt.subplots(1, 2, sharey=True)

for col, task in enumerate(['COMBINED_BINARY_AS_MULTI', 'COMBINED_MULTI']):
    violindict = tasks[task]['Violin']

    for i in range(2):
        arrs = []
        labels = [""]
        for n, dset in enumerate(violindict):
            array = np.array(violindict[dset])[:,i]
            arrs.append(sorted(array[~np.isnan(array)]))
            labels.append(dset)

        stats = np.array([np.percentile(arr, [25, 50, 75]) for arr in arrs])
        quartile1 = stats[:,0]
        medians = stats[:,1]
        quartile3 = stats[:,2]
        inds = np.arange(1, len(medians) + 1)
        whiskers = np.array([
            adjacent_values(sorted_array, q1, q3)
            for sorted_array, q1, q3 in zip(arrs, quartile1, quartile3)])
        whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

        for idx, arr in enumerate(arrs):
            if len(arr) > 10:
                arr = np.where((arr >= whiskers_min[idx]) & (arr <= whiskers_max[idx]), arr, np.nan)
                arr = arr[~np.isnan(arr)]
                arrs[idx] = arr
        if i == 0:
            ax = ax1[col]
        else:
            ax = ax2[col]

        
        plot = ax.violinplot(arrs, points=60, showextrema=False, bw_method=.3)
        # fill with colors
        colors = ['firebrick', 'firebrick', 'seagreen', 'navy', 'navy', 'navy']
        for patch, color in zip(plot['bodies'], colors):
            patch.set_alpha(0.20)
            patch.set_facecolor(color)
            
        ax.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
        ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
        ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylim(-0.01, 1.1)
        ax.set_xlabel('Cohort')
        if col == 0:
            ax.set_ylabel(ylabels[i])


# %%
# Create merged violinplots
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 26


ylabels = ['Dice Similarity Coefficient',
           'Lesion Detection F1']

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value

fig1, ax1 = plt.subplots(figsize=(15, 7.5))
fig2, ax2 = plt.subplots(figsize=(15, 7.5))

for col, task in enumerate(['COMBINED_BINARY_AS_MULTI', 'COMBINED_MULTI']):
    violindict = tasks[task]['Violin']
    violindict2 = {'MS':violindict['MS08']+violindict['MSSEG'],
    'Stroke': violindict['Stroke'], 
    'WMH':violindict['WMH - A']+violindict['WMH - S']+violindict['WMH - U']}
    for i in range(2):
        arrs = []
        labels = []
        for n, dset in enumerate(violindict2):
            array = np.array(violindict2[dset])[:,i]
            arrs.append(sorted(array[~np.isnan(array)]))
            labels.append(dset)

        stats = np.array([np.percentile(arr, [25, 50, 75]) for arr in arrs])
        quartile1 = stats[:,0]
        medians = stats[:,1]
        quartile3 = stats[:,2]
        inds = np.arange(col+1, len(medians)*3, 3)
        whiskers = np.array([
            adjacent_values(sorted_array, q1, q3)
            for sorted_array, q1, q3 in zip(arrs, quartile1, quartile3)])
        whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

        #for idx, arr in enumerate(arrs):
        #    if len(arr) > 10:
        #        arr = np.where((arr >= whiskers_min[idx]) & (arr <= whiskers_max[idx]), arr, np.nan)
        #        arr = arr[~np.isnan(arr)]
        #        arrs[idx] = arr
        if i == 0:
            ax = ax1
            lab = 'Cascade'
        else:
            ax = ax2
            lab = 'Multiclass'

        colors = ['seagreen', 'seagreen', 'seagreen']
        if col == 1:
            colors = ['navy', 'navy', 'navy']

        
        plot = ax.violinplot(arrs, positions=inds, points=60, widths=1,showextrema=False, bw_method=.3)
        # fill with colors
        for patch, color in zip(plot['bodies'], colors):
            patch.set_alpha(0.20)
            patch.set_facecolor(color)

        ax.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
        ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
        ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)
        ax.set_xticks(np.arange(1.5, 9, 3))
        ax.set_xticklabels(labels)
        ax.set_ylim(-0.01, 1.1)
        if col == 0:
            ax.set_ylabel(ylabels[i])
        ax.legend(['Cascade', 'Multiclass'], loc='upper left')
        leg = ax.get_legend()
        leg.legend_handles[0].set_color('seagreen')
        leg.legend_handles[0].set_alpha(0.20)
        leg.legend_handles[1].set_color('navy')
        leg.legend_handles[1].set_alpha(0.20)

fig1.savefig('/home/zcr545/DSC.pdf', bbox_inches='tight', dpi=1400)
fig2.savefig('/home/zcr545/OBJF1.pdf', bbox_inches='tight', dpi=1400)




# %%
# Get STD for missings tasks

MSSEG_binary = '/home/zcr545/YuccaData/yucca_segmentations/Task054_MSSeg/Task052_3BrainLesionFTBalanced/UNet2D/YuccaTrainerV2_FT__YuccaPlannerV2Z/fold_0_checkpoint_best/results.json'
MSSEG_spec = '/home/zcr545/YuccaData/yucca_segmentations/Task054_MSSeg/Task054_MSSeg/UNet2D/YuccaTrainerV2__YuccaPlannerV2Z/fold_0_checkpoint_best/results.json'

MS_binary = '/home/zcr545/YuccaData/yucca_segmentations/Task041_MSLesion08/Task052_3BrainLesionFTBalanced/UNet2D/YuccaTrainerV2_FT__YuccaPlannerV2Z/fold_0_checkpoint_best/results.json'
MS_spec = '/home/zcr545/YuccaData/yucca_segmentations/Task041_MSLesion08/Task041_MSLesion08/UNet2D/YuccaTrainerV2__YuccaPlannerV2Z/fold_0_checkpoint_best/results.json'

Stroke_binary = '/home/zcr545/YuccaData/yucca_segmentations/Task042_ISLES22/Task052_3BrainLesionFTBalanced/UNet2D/YuccaTrainerV2_FT__YuccaPlannerV2Z/fold_0_checkpoint_best/results.json'
Stroke_spec = '/home/zcr545/YuccaData/yucca_segmentations/Task042_ISLES22/Task042_ISLES22/UNet2D/YuccaTrainerV2__YuccaPlannerV2Z/fold_0_checkpoint_best/results.json'

WMH_binary = '/home/zcr545/YuccaData/yucca_segmentations/Task003_WMH_Flair/Task052_3BrainLesionFTBalanced/UNet2D/YuccaTrainerV2_FT__YuccaPlannerV2Z/fold_0_checkpoint_best/results.json'
WMH_spec = '/home/zcr545/YuccaData/yucca_segmentations/Task003_WMH_Flair/Task003_WMH_Flair/UNet2D/YuccaTrainerV2__YuccaPlannerV2Z/fold_0_checkpoint_best/results.json'

for i in [MSSEG_spec, MSSEG_binary, MS_spec,MS_binary, Stroke_spec,Stroke_binary,WMH_spec,WMH_binary]:
    file = load_json(i)
    dice = []
    vs = []
    for sub in file:
        if not sub == "mean":
            dice.append(file[sub]["1"]['Dice'])
            vs.append(file[sub]["1"]['Volume Similarity'])
    print(round(np.nanmean(dice), 3), round(np.nanstd(dice), 2), round(np.nanmean(vs), 3), round(np.nanstd(vs), 2))
# %%
