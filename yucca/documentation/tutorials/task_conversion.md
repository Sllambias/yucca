# Task Conversion

Yucca uses the data structure presented in the Medical Segmentation Challenge [see @https://arxiv.org/pdf/1902.09063.pdf; https://arxiv.org/pdf/1902.09063.pdf]. This includes naming conventions, directory structures and descriptor files (dataset.json).

# File and Folder Structure
Yucca reads/writes data from/to four directories specified by the user where the directories contain respectively `raw_data`, `preprocessed_data`, `models` and `results`. The paths to these directories should be specified in your environment variables as seen [here](/yucca/documentation/tutorials/environment_variables.md).

The raw data used by Yucca is expected to have a specific format. To convert a dataset (e.g. some dataset you have downloaded) to Yucca-compliant raw data adhere to the guidelines below:

### The `raw_data` Directory
Yucca expects raw data to be located in subdirectories of the `raw_data` directory. The name of the subdirectory for a given dataset should be its Task Name (see below).

### Task Names 
Datasets must be assigned a Task Name of the format `TaskXXX_MYTASK` where `XXX` is a unique 3-digit identifier and `MYTASK` is a freely chosen dataset name. 
For instance, the OASIS hippocampus segmentation dataset is called `Task001_OASIS`. Which means that Yucca will assume the raw data is found at; `raw_data/Task001_OASIS`

### Train/Test Split
Inside the task directory (e.g. `raw_data/Task001_OASIS`) data should be split into Train and Test splits (subdirectories) in almost all cases. However, to completely avoid any data leakage, Yucca will NOT do this for you. To ensure a clear division between training and testing data Yucca will automatically only train on data placed in subdirectories called `imagesTr` (training). Therefore, you have to save training and testing samples in appropriately named directories. Training images and labels should be in placed in the subdirectories `imagesTr` and `labelsTr` respectively. And, if they exist, testing images and labels should be placed in the subdirectories `imagesTs` and `labelsTs` respectively.

### File Names
Image files must be named according to the format `ID_MODALITY.nii.gz` where `ID` is replaced by e.g. `sub_01` and `MODALITY` is replaced by the appropriate modality identifier.
If the dataset only contains a single modality (e.g. T1 MR images) all files are simply suffixed with the same modality identifier `000`.
If the dataset contains multiple modalities each sequence should be assigned a unique modality suffix. For example, for a dataset containining T1 and CT images, the T1 images could be assigned the `000` suffix and the CT images could be suffixed by `001`.
Labels are stored WITHOUT the modality identifier.

### Example Directory Structure (for a multimodal dataset):
This is how the contents of your `raw_data` folder could look:
- Note that we still only have 1 label (segmentation) per subject, even though each subject may have multiple images (e.g. both CT and MRI scans).
```
raw_data/
├── Task001_OASIS/
|   ├── imagesTr/
|   |   ├── sub_01_000.nii.gz
|   |   ├── sub_01_001.nii.gz
|   |   ├── sub_04_000.nii.gz
|   |   ├── sub_04_001.nii.gz
|   |   ├── ...
|   ├── labelsTr/
|   |   ├── sub_01.nii.gz
|   |   ├── sub_04.nii.gz
|   |   ├── ...
|   ├── imagesTs/
|   |   ├── sub_02_000.nii.gz
|   |   ├── sub_02_001.nii.gz
|   |   ├── sub_03_000.nii.gz
|   |   ├── sub_03_001.nii.gz
|   |   ├── ...
|   ├── labelsTs/
|   |   ├── sub_02.nii.gz
|   |   ├── sub_03.nii.gz
|   |   ├── ...
├── Task002_OtherTask/
```

### Task Conversion Scripts
Task conversion can be performed by running the [`run_task_conversion.py`](/yucca/yucca/run/run_task_conversion.py) which executes a task conversion according to one of the files in `yucca/yucca/task_conversion`. For some tasks a task conversion file already exists (e.g. [`Task001_OASIS`](/yucca/yucca/task_conversion/Task001_OASIS.py)) but otherwise a task conversion file must be created. A template for task conversion files can be found [here](/yucca/yucca/task_conversion/template.py).


# Preprocessing

**Registration**:
Each image(s)/label pair must be properly registered and shaped. This implies identical orientation, spacing and size.

**Labels**:
Ensure that only the correct labels are present in the ground truth segmentations.

In some cases each token of a given type is assigned a unique label. E.g. in a microbleed segmentation/detection task each bleed in an image is labeled with incrementing integers. This is often highly undesirable as the segmentation network will treat each label as unique types rather than tokens of the same type.

Changing all non-background labels to a single label can be achieved using:
```
# SITK METHOD
      import SimpleITK as sitk
      segmentation = sitk.ReadImage('path/to/seg01.nii.gz')
      label = sitk.Mask(label, sitk.Not(label != 0), 1)

# NIBABEL METHOD
      import nibabel as nib
      segmentation = nib.load('path/to/seg01.nii.gz')
      data = segmentation.get_fdata()
      data[data != 0] = 1
      segmentation = nib.Nifti1Image(data, segmentation.affine, segmentation.header)
```
However, beware to not use this in cases where it is desirable to have multiple labels (e.g. for a left and right hippocampus or white and gray matter)
