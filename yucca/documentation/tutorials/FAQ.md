# Out of Memory errors (OOM)

1. Check `keep_in_ram` is disabled in the [`YuccaTrainDataset`](/yucca/yucca/training/data_loading/YuccaDataset.py). This setting should only be enabled for smaller datasets as it will quickly results in OOM errors for large datasets.
