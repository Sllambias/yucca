# Task Conversion

Yucca uses the data structure presented in the [Medical Segmentation Challenge](https://arxiv.org/pdf/1902.09063.pdf). This includes naming conventions, directory structures and descriptor files (dataset.json).

# File and Folder Structure
Yucca reads/writes data from/to four directories specified by the user where the directories contain respectively `raw_data`, `preprocessed_data`, `models` and `results`. The paths to these directories should be specified in your environment variables as seen [here](/yucca/documentation/guides/environment_variables.md).

The raw data used by Yucca is expected to have a specific format. To convert a dataset (e.g. some dataset you have downloaded) to Yucca-compliant raw data adhere to the guidelines below:

### The `raw_data` Directory
Yucca expects raw data to be located in subdirectories of the `raw_data` directory. The name of the subdirectory for a given dataset should be its Task Name (see below).

### Task Names
Datasets must be assigned a Task Name of the format `TaskXXX_MYTASK` where `XXX` is a unique 3-digit identifier and `MYTASK` is a freely chosen dataset name.
For instance, the OASIS hippocampus segmentation dataset is called `Task001_OASIS`. Which means that Yucca will assume raw data can be found at; `raw_data/Task001_OASIS`

### Train/Test Split
Inside the task directory (e.g. `raw_data/Task001_OASIS`) data should be split into Train and Test splits (subdirectories) in almost all cases. To completely avoid any data leakage, Yucca will NOT do this for you. To ensure a clear division between training and testing data Yucca will automatically only train on data placed in subdirectories called `imagesTr` (training). Therefore, you have to save training and testing samples in appropriately named directories. Training images and labels should be in placed in the subdirectories `imagesTr` and `labelsTr` respectively. And, if they exist, testing images and labels should be placed in the subdirectories `imagesTs` and `labelsTs` respectively.

### File Names
Image files must be named according to the format `ID_MODALITY.EXTENSION`.
- `ID` is the unique case identifier, such as `sub_01` or `image_4125`.
- `MODALITY` is a 3 digit suffix that identifies the modality of an **image**. For example, the imaging technique (e.g. T1, CT or ultrasound) or timepoint. Yucca uses this information to (1) normalize each modality individually and (2) concatenate cases (in the channel dimension) that have multiple images. If the dataset only contains one modality (or you wish to treat all modalities as one) simply suffix all images with the modality identifier `000` - this is generally the approach. If the dataset contains multiple modalities each should be assigned a unique modality suffix. For example, for a medical imaging dataset containing T1 and CT images for all labels, the T1 images could be assigned by `000` and the CT images could be suffixed by `001`. Note that any **labels** are stored without modality identifiers.
- `EXTENSION` is any of the supported file extensions. Currently `.nii.gz` and `.png` is supported for the imagesTr and ImagesTs folders. For the labelsTr and labelsTs folders `.nii.gz`, `.png` and `.txt` are supported.

### Example Directory Structure for a multimodal segmentation dataset:
This is how the contents of your `raw_data` folder could look:
- Note that we still only have 1 label (segmentation) per subject, even though each subject may have multiple images (e.g. both CT and MRI scans).
- If this was a classification dataset the labelsTr and labelsTs folders should be populated with `.txt` files containing the class(es) of the case.
- If this was an unsupervised dataset the labelsTr and labelsTs folder should be empty.
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
Task conversion can be performed by running the [`run_task_conversion.py`](/yucca/run/run_task_conversion.py) which executes a task conversion according to one of the files in `yucca/yucca/task_conversion`. For some tasks a task conversion file already exists (e.g. [`Task001_OASIS`](/yucca/task_conversion/Task001_OASIS.py)) but otherwise a task conversion file must be created. A template for task conversion files can be found [here](/yucca/task_conversion/template.py).

# Preprocessing

**Registration**:
Each image(s)/label pair must be properly registered and shaped. This implies identical orientation, spacing and size.

**Labels**:
Ensure that only the correct labels are present in the ground truth segmentations.

In some cases each token of a given type is assigned a unique label. E.g. in a microbleed segmentation/detection task each bleed in an image is labeled with incrementing integers. This is often highly undesirable as the segmentation network will treat each label as unique types rather than tokens of the same type.
