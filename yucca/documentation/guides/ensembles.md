## Ensembles

To train an ensemble of models we use the `yucca_preprocess`, `yucca_train` and `yucca_inference` commands. For advanced usage see: [`run_scripts_advanced.py`](yucca/documentation/guides/run_scripts_advanced.md#ensembles). A common application of model ensembles is to train 2D models on each of the three axes of 3D data (either denoted as the X-, Y- and Z-axis or, in medical imaging, the axial, sagittal and coronal views) and then fuse their predictions in inference. 

To train 3 models on the three axes of a 3D dataset called `Task001_Brains` prepare three preprocessed versions of the dataset using the three Planners `YuccaPlannerX`, `YuccaPlannerY` and `YuccaPlannerZ`:
```console
> yucca_preprocess -t Task001_Brains -pl YuccaPlannerX
> yucca_preprocess -t Task001_Brains -pl YuccaPlannerY
> yucca_preprocess -t Task001_Brains -pl YuccaPlannerZ
```

Then, train three 2D models one on each version of the preprocessed dataset:
```console
> yucca_train -t Task001_Brains -pl YuccaPlannerX -d 2D
> yucca_train -t Task001_Brains -pl YuccaPlannerY -d 2D
> yucca_train -t Task001_Brains -pl YuccaPlannerZ -d 2D
```

Then, run inference on the target dataset with each trained model.
```console
> yucca_inference -t Task001_Brains -pl YuccaPlannerX -d 2D
> yucca_inference -t Task001_Brains -pl YuccaPlannerY -d 2D
> yucca_inference -t Task001_Brains -pl YuccaPlannerZ -d 2D
```

Finally, fuse their results and evaluate the predictions.
```console
> yucca_ensemble --in_dirs /path/to/predictionsX /path/to/predictionsY /path/to/predictionsZ --out_dir /path/to/ensemble_predictionsXYZ
```
