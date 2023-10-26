#%%
from batchgenerators.utilities.file_and_folder_operations import subfiles, join
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
#plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 22

intensity_dict = {'MS': [],
                  'Stroke': [],
                  'WMH': []}

base = '/home/zcr545/YuccaData/yucca_preprocessed/Task052_3BrainLesionFTBalanced/YuccaPlannerV2X'
for seg in subfiles(base, suffix='.npy'):
    if 'Center' in seg or 'CHB' in seg or 'UNC' in seg:
        dataset = 'MS'
    elif 'stroke' in seg:
        dataset = 'Stroke'
    else:
        dataset = 'WMH'
    
    if seg[-9] == '_':
        if not seg[-8] == '0':
            continue
    data = np.load(seg)
    imarr = data[0]
    segarr = data[-1]

    f = imarr*segarr
    f = f[f>0]
    intensity_dict[dataset].append(f.flatten())

for dataset in intensity_dict.keys():
    intensity_dict[dataset] = sum([i.astype(float).tolist() for i in intensity_dict[dataset] if len(i) > 0], [])

# %%
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

colors = ['green','sienna','dodgerblue']
for i, dataset in enumerate(list(intensity_dict.keys())):
    plt.rcParams['font.size'] = 14

    plot = sns.kdeplot(np.array(intensity_dict[dataset], dtype=float)[::20], color=colors[i], bw_adjust=6, clip=[-4,12], fill=True)
    sns.despine(bottom = True, left = True)
    labels = [round(i, 1) for i in np.arange(0., 1.1, 0.1)]
    plot.set_xticks(np.linspace(-4, 12, 11), labels=labels, fontsize=12)
    plot.tick_params(bottom=False, left=False)
    plt.legend(title='Dataset', loc='upper right', labels=list(intensity_dict.keys()))
    plt.rcParams['font.size'] = 22
    plot.set_xlabel("Normalized Intensities")
    plot.set_ylabel("Probability")

plot.figure.savefig('/home/zcr545/DF.pdf', bbox_inches='tight', dpi=1400)


# %%
