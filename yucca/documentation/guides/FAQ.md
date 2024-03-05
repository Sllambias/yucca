# Out of Memory errors (OOM)
## During Training
1. Verify that `keep_in_ram` is disabled in the [`YuccaTrainDataset`](/yucca/training/data_loading/YuccaDataset.py). This setting should only be enabled for smaller datasets as it will quickly results in OOM errors for large datasets.
2. Reduce the number of workers used in the DataLoaders.
3. Increase the number of CPU's available.

## During Inference
1. Request more ram per CPU if you're working with large arrays.
