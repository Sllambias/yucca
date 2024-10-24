## Label Regions

What are regions?
Regions are hierarchical groups of labels. Taking BRATS21 as an example, there are 4 labels: background (BG), Necrotic Tumor Core (NCR), peritumoral edematous/invaded tissue (ED), GD-enhancing tumor (ET). These are grouped in 3 regions:  Whole Tumor (WT) consisting of [ET, NCR, ED], Tumor Core (TC) consisting of [ET, NCR] and Enhancing Tumor (ET) consisting of [ET].

Regions are distinct from treating WT, TC and ET as 3 regular labels. Normally, if the model predicted ET in the TC area, it would be penalized, as pixels/voxels/samples are assigned exactly one class for multiclass segmentation. Regions however are multilabel, and this means the model can be *partially* correct. For the sake of evaluating TC all positive ET predictions are included. Even though the model mistakenly identified the voxel as Enhancing Tumor, it is still rewarded for correctly identifying the Tumor Core. 

In practice this is implemented by defining the labels from the regions in the Task Conversion step:

```
labels={0: "BG", 1: "NCR", 2: "ED", 3: "ET"},
regions={
    "WT": {"priority": 3, "labels": ["ET", "NCR", "ED"]},
    "TC": {"priority": 2, "labels": ["ET", "NCR"]},
    "ET": {"priority": 1, "labels": ["ET"]},
},
```

The priorities defines the order in which we "collapse" them, if we wish to do that after inference.
E.g. going from a 3-channel prediction with one channel per region, we can go to a regular 1-channel prediction by first converting the lowest priority region into label *n*, and then overlaying the following regions with *n* increasing by 1 for each region, so we end up with WT = 1, TC = 2, and ET = 3.

To train a model using regions simply employ a manager with *self.use_label_regions = True*. 

In the pipeline this will change the loss to SigmoidDiceBCE and enable the ConvertLabelsToRegions transform, which pulls the region information from the dataset.json (with the regions you defined during task conversion).

During inference the arrays are saved as multi-channel predictions with the regions, and the ground truth is converted from labels to regions for evaluation.