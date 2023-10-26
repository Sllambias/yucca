| Task | Dataset | Modality | Dice | Network | Trainer | Planner | Notes | User | Source(s) | 
|---|---|---|---|---|---|---|---|---|---|
| Hippocampus | OASIS | T1 | 0.86/0.85 | 2D UNet | YuccaTrainer | YuccaPlansY | | [Llambias](https://github.com/Sllambias) | http://www.neuromorphometrics.com/2012_MICCAI_Challenge_Data.html; https://www.oasis-brains.org/ |
| Hippocampus | OASIS | T1 | 0.84/0.84 | 2D UNet | YuccaTrainer | YuccaPlansZ | | [Llambias](https://github.com/Sllambias) | http://www.neuromorphometrics.com/2012_MICCAI_Challenge_Data.html; https://www.oasis-brains.org/ |
| Hippocampus | OASIS | T1 | 0.62/0.63 | 2D UNet | YuccaTrainer | YuccaPlans | | [Llambias](https://github.com/Sllambias) | http://www.neuromorphometrics.com/2012_MICCAI_Challenge_Data.html; https://www.oasis-brains.org/ |
| Hippocampus | OASIS | T1 | 0.87/0.87 | 3D UNet | YuccaTrainer | YuccaPlans | | [Llambias](https://github.com/Sllambias) | http://www.neuromorphometrics.com/2012_MICCAI_Challenge_Data.html; https://www.oasis-brains.org/ |
| Hippocampus | LPBA40 | T1 | 0.86/0.85 | 3D UNet | YuccaTrainer | YuccaPlans | | [Llambias](https://github.com/Sllambias) | https://www.sciencedirect.com/science/article/pii/S1053811907008099 |
| Hippocampus | HarP | T1 | 0.87/0.87 | 3D MultiResUNet | YuccaTrainer | YuccaPlans | | [Llambias](https://github.com/Sllambias) | https://www.sciencedirect.com/science/article/pii/S155252601402891X |
| Hippocampus | HarP | T1 | 0.88/0.88 | 3D UNet | YuccaTrainer | YuccaPlans | | [Llambias](https://github.com/Sllambias) | https://www.sciencedirect.com/science/article/pii/S155252601402891X |
| Hippocampus | Hammers | T1 | 0.80/0.83 | 3D UNet | YuccaTrainer | YuccaPlans | | [Llambias](https://github.com/Sllambias) | www.brain-development.org; https://www.sciencedirect.com/science/article/pii/S1053811917300964; https://www.sciencedirect.com/science/article/pii/S1053811907010634; https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6871794/ |
| White Matter Hyperintensity | WMH MICCAI '17 | Flair/T1 | 0.76 | 2D UNet | YuccaTrainerV2 | YuccaPlannerV2Z | | [Llambias](https://github.com/Sllambias) | https://wmh.isi.uu.nl/results/results-miccai-2017/; https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7590957/ |
| Breast Tumor | USBreast | Ultrasound | ? | ? | ? | ? | |[Llambias](https://github.com/Sllambias) | https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset |
| Brain Tumour | Decathlon | Flair/T1/T1gd/T2 | 0.77/0.56/0.69 | 3D UNet | YuccaTrainerV2 | YuccaPlans |[See: Note A](/yucca/documentation/results.md#1-test-samples-not-available) | [Llambias](https://github.com/Sllambias) |  http://medicaldecathlon.com/ |
| Heart | Decathlon | T1 | 0.90 | 3D UNet | YuccaTrainerV2 | YuccaPlans | [See: Note A](/yucca/documentation/results.md#1-test-samples-not-available) |[Llambias](https://github.com/Sllambias) |  http://medicaldecathlon.com/ |
| Liver | Decathlon | CT | 0.95/0.49 | 3D UNet | YuccaTrainerV2 | YuccaPlans |[See: Note A](/yucca/documentation/results.md#1-test-samples-not-available) | [Llambias](https://github.com/Sllambias) |  http://medicaldecathlon.com/ |
| Hippocampus | Decathlon | MRI | 0.87/0.85 | 3D UNet | YuccaTrainer | YuccaPlans | [See: Note A](/yucca/documentation/results.md#1-test-samples-not-available)| [Enevoldsen](https://github.com/Drilip) | http://medicaldecathlon.com/ |
| Prostate | Decathlon | T2/ADC | 0.50/0.70 | 3D UNet | YuccaTrainerV2 | YuccaPlans | [See: Note A](/yucca/documentation/results.md#1-test-samples-not-available)| [Llambias](https://github.com/Sllambias) |  http://medicaldecathlon.com/ |
| Prostate | Decathlon | T2/ADC | 0.50/0.64 | 2D UNet | YuccaTrainer | YuccaPlansZ | [See: Note A](/yucca/documentation/results.md#1-test-samples-not-available)| [Llambias](https://github.com/Sllambias) |  http://medicaldecathlon.com/ |
| Lung | Decathlon | CT | 0.65 | 3D UNet | YuccaTrainer | YuccaPlans |[See: Note A](/yucca/documentation/results.md#1-test-samples-not-available) | [Enevoldsen](https://github.com/Drilip) |  http://medicaldecathlon.com/ |
| Pancreas | Decathlon | CT | 0.78/0.30 | 3D UNet | YuccaTrainerV2 | YuccaPlans | [See: Note A](/yucca/documentation/results.md#1-test-samples-not-available)| [Llambias](https://github.com/Sllambias) |  http://medicaldecathlon.com/ |
| Hepatic Vessel | Decathlon | CT | 0.50/0.37 | 2D UNet | YuccaTrainerV2 | YuccaPlannerV2 |[See: Note A](/yucca/documentation/results.md#1-test-samples-not-available) | [Llambias](https://github.com/Sllambias) | http://medicaldecathlon.com/ |
| Spleen | Decathlon | CT  | 0.90 | 3D UNet | YuccaTrainer | YuccaPlans | [See: Note A](/yucca/documentation/results.md#1-test-samples-not-available)| [Enevoldsen](https://github.com/Drilip) | http://medicaldecathlon.com/ |
| Colon | Decathlon | CT  | ? | ? | ? | ? |[See: Note A](/yucca/documentation/results.md#1-test-samples-not-available) | [Llambias](https://github.com/Sllambias) |  http://medicaldecathlon.com/ |
| MS Lesion | MS Lesion '08 | MRI | 0.35 | 2D UNet Ensemble | YuccaTrainerV2 | YuccaPlannerV2 | [See: Note A](/yucca/documentation/results.md#1-test-samples-not-available) | [Llambias](https://github.com/Sllambias) |  http://www.ia.unc.edu/MSseg/index.html |
| Stroke Lesion | ISLES22 | DWI | 0.70 | 2D UNet Ensemble | YuccaTrainerV2 | YuccaPlannerV2 | Evaluated using the [challenge implementation](/yucca/evaluation/challenge_evaluation_scripts/isles_eval.py) | [Llambias](https://github.com/Sllambias) |  https://isles22.grand-challenge.org/ |




# A: Test samples not available
For various reasons, test samples may not be available. E.g. challenges are often discontinued or abandoned while only training labels are publicly available. 
This means we have to create a test set from the available (training) samples. Therefore the results may not be comparable to historical results posted on challenge websites or in research papers. In some cases our results may be better, because the original test set included a domain shift not represented in the training set. However, in most cases where the test set is unavailable the performance of our models will be poorer, because the size of the training set is reduced significantly.  
