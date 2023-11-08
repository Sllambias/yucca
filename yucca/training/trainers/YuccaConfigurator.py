
import torch
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, load_json, load_pickle
from yucca.paths import yucca_models, yucca_preprocessed
from yuccalib.image_processing.matrix_ops import get_max_rotated_size
from yuccalib.network_architectures.utils.model_memory_estimation import find_optimal_tensor_dims


class YuccaConfigurator:
    def __init__(
            self,
            folds: int = 0,
            max_vram: int = 12,
            model_dimensions: str = '3D',
            model_name: str = 'UNet',
            planner: str = 'YuccaPlanner',
            task: str = None,

            ):
        self.folds = folds
        self.model_dimensions = model_dimensions
        self.model_name = model_name
        self.name = self.__class__.__name__
        self.planner = planner
        self.task = task
        self.max_vram = max_vram
        self.run_setup()

    def run_setup(self):
        self.setup_paths_and_plans()
        self.setup_splits()
        self.setup_train_params()
        print(
            f"Using training data from: {self.train_data_dir} \n"
            f"Saving model outputs in: {self.outpath} \n"
            f"Using patch size: {self.patch_size} \n"
            f"Using initial patch size: {self.initial_patch_size} \n"
            f"Using batch size: {self.batch_size} \n"
        )

    def setup_paths_and_plans(self):
        self.train_data_dir = join(yucca_preprocessed, self.task, self.planner)

        self.outpath = join(
            yucca_models,
            self.task,
            self.model_name + '__' + self.planner,
            self.model_dimensions,
            self.name, str(self.folds))

        maybe_mkdir_p(self.outpath)

        self.plans_path = join(
            yucca_preprocessed, 
            self.task,
            self.planner,
            self.planner + '_plans.json')

        self.plans = load_json(self.plans_path)

    def setup_splits(self):
        splits_file = load_pickle(join(yucca_preprocessed, self.task, 'splits.pkl'))
        self.train_split = splits_file[self.folds]['train']
        self.val_split = splits_file[self.folds]['val']

    def setup_train_params(self):
        self.num_classes = len(self.plans['dataset_properties']['classes'])
        self.num_modalities = len(self.plans['dataset_properties']['modalities'])
        if torch.cuda.is_available():
            self.batch_size, self.patch_size = find_optimal_tensor_dims(dimensionality=self.model_dimensions,
                                                                        num_classes=self.num_classes,
                                                                        modalities=self.num_modalities,
                                                                        model_name=self.model_name,
                                                                        max_patch_size=self.plans['new_mean_size'],
                                                                        max_memory_usage_in_gb=self.max_vram)
        else:
            print("Cuda is not available, using tiny patch and batch")
            self.batch_size = 2
            self.patch_size = (32, 32, 32)
        self.initial_patch_size = get_max_rotated_size(self.patch_size)
