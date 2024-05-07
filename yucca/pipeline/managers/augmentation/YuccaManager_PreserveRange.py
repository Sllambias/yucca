from yucca.pipeline.managers.YuccaManager import YuccaManager


class YuccaManager_PreserveRange(YuccaManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmentation_params["clip_to_input_range"] = True


if __name__ == "__main__":
    # Test that shows the difference between no clipping and clipping to input range.
    # Relevant for models trained on very strict ranges.
    man = YuccaManager_PreserveRange(
        task="Task001_OASIS",
        planner="YuccaPlanner_1_1_1",
        enable_logging=False,
        model_dimensions="2D",
        patch_size=(64, 64),
        batch_size=2,
    )
    man.initialize(stage="fit")
    man.data_module.setup(stage="fit")
    dataset = iter(man.data_module.train_dataloader())
    print("All samples start in the range 0-1. \n Testing the YuccaManager_PreserveRange")
    for i in range(20):
        sample = next(dataset)
        print(f'Sample {i} min value: {sample["image"].min()} and max value: {sample["image"].max()}')
