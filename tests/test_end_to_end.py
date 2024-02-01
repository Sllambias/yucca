import os


def test_task_convert():
    os.system("yucca_convert_task -t Task000_TEST")
    os.system("yucca_convert_task -t Task000_TEST_CLASSIFICATION")


def test_preprocessing():
    from yucca.paths import yucca_preprocessed_data

    os.system("yucca_preprocess -t Task000_TEST")
    assert len(os.listdir(os.path.join(yucca_preprocessed_data, "Task000_TEST", "YuccaPlanner"))) > 0

    os.system("yucca_preprocess -t Task000_TEST -pl UnsupervisedPlanner")
    assert len(os.listdir(os.path.join(yucca_preprocessed_data, "Task000_TEST", "UnsupervisedPlanner"))) > 0

    os.system("yucca_preprocess -t Task000_TEST_CLASSIFICATION -pl YuccaPlanner_MaxSize -pr ClassificationPreprocessor")
    assert len(os.listdir(os.path.join(yucca_preprocessed_data, "Task000_TEST_CLASSIFICATION", "YuccaPlanner_MaxSize"))) > 0


def test_training():
    from yucca.paths import yucca_models

    # First: a very basic short training
    # os.system(
    #    "yucca_train -t Task000_TEST -m TinyUNet --epochs 2 --batch_size 2 --train_batches_per_step 5 --val_batches_per_step 5"
    # )

    # Second: a training with a lot of changed parameters
    os.system(
        "yucca_train -t Task000_TEST -m TinyUNet -d 2D -pl UnsupervisedPlanner --lr 0.0006 --loss CE --mom 0.99 "
        "--new_version --profile --experiment NonDefault --patch_size 32 --split_idx 0 --split_data_param 0.7 "
        "--split_data_method simple_train_val_split --epochs 2 --batch_size 2 --train_batches_per_step 5 --val_batches_per_step 5"
    )
    expected_outpath = os.path.join(
        yucca_models,
        "Task000_TEST",
        "TinyUNet__2D",
        "YuccaManager__UnsupervisedPlanner",
        "NonDefault",
        "simple_train_val_split_7_fold_0/checkpoints",
    )
    assert len(os.listdir(expected_outpath)) > 0
