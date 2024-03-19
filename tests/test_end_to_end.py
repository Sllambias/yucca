import os
import subprocess


def test_task_convert():
    subprocess.run(["yucca_convert_task", "-t", "Task000_TEST_SEGMENTATION"], check=True)
    subprocess.run(["yucca_convert_task", "-t", "Task000_TEST_CLASSIFICATION"], check=True)
    subprocess.run(["yucca_convert_task", "-t", "Task000_TEST_UNSUPERVISED"], check=True)


def test_preprocessing():
    subprocess.run(["yucca_preprocess", "-t", "Task000_TEST_SEGMENTATION"], check=True)
    subprocess.run(["yucca_preprocess", "-t", "Task000_TEST_UNSUPERVISED", "-pl", "UnsupervisedPlanner"], check=True)
    subprocess.run(
        [
            "yucca_preprocess",
            "-t",
            "Task000_TEST_CLASSIFICATION",
            "-pl",
            "YuccaPlanner_MaxSize",
            "-pr",
            "ClassificationPreprocessor",
        ],
        check=True,
    )


def test_training():
    # First: a very basic short training
    subprocess.run(
        [
            "yucca_train",
            "-t",
            "Task000_TEST_SEGMENTATION",
            "-m",
            "TinyUNet",
            "--epochs",
            "2",
            "--batch_size",
            "2",
            "--disable_logging",
            "--train_batches_per_step",
            "3",
            "--val_batches_per_step",
            "3",
        ],
        check=True,
    )

    # First: a very basic short training
    subprocess.run(
        [
            "yucca_train",
            "-t",
            "Task000_TEST_SEGMENTATION",
            "-m",
            "TinyUNet",
            "-man",
            "YuccaManager_AllAlways",
            "--epochs",
            "2",
            "--batch_size",
            "2",
            "--patch_size",
            "32",
            "--disable_logging",
            "--train_batches_per_step",
            "1",
            "--val_batches_per_step",
            "1",
        ],
        check=True,
    )
def test_finetune():
    from yucca.paths import yucca_models

    chk = os.path.join(
        yucca_models,
        "Task000_TEST_SEGMENTATION",
        "TinyUNet__3D",
        "YuccaManager__YuccaPlanner",
        "default",
        "kfold_5_fold_0",
        "version_0",
        "checkpoints",
        "best.ckpt",
    )
    # Second: a training with a lot of changed parameters
    subprocess.run(
        [
            "yucca_finetune",
            "-t",
            "Task000_TEST_UNSUPERVISED",
            "-chk",
            chk,
            "-m",
            "TinyUNet",
            "-d",
            "2D",
            "-pl",
            "UnsupervisedPlanner",
            "--lr",
            "0.0006",
            "--mom",
            "0.99",
            "--disable_logging",
            "--new_version",
            "--profile",
            "--experiment",
            "NonDefault",
            "--patch_size",
            "32",
            "--split_idx",
            "0",
            "--split_data_param",
            "0.7",
            "--split_data_method",
            "simple_train_val_split",
            "--epochs",
            "2",
            "--batch_size",
            "2",
            "--train_batches_per_step",
            "2",
            "--val_batches_per_step",
            "2",
        ],
        check=True,
    )


def test_inference():
    subprocess.run(
        [
            "yucca_inference",
            "-s",
            "Task000_TEST_SEGMENTATION",
            "-t",
            "Task000_TEST_SEGMENTATION",
            "-d",
            "3D",
            "-m",
            "TinyUNet",
            "--no_wandb",
        ],
        check=True,
    )


def test_evaluation():
    subprocess.run(
        [
            "yucca_evaluation",
            "-s",
            "Task000_TEST_SEGMENTATION",
            "-t",
            "Task000_TEST_SEGMENTATION",
            "-d",
            "3D",
            "-m",
            "TinyUNet",
            "--no_wandb",
            "--surface_eval",
        ],
        check=True,
    )

    subprocess.run(
        [
            "yucca_evaluation",
            "-s",
            "Task000_TEST_SEGMENTATION",
            "-t",
            "Task000_TEST_SEGMENTATION",
            "-d",
            "3D",
            "-m",
            "TinyUNet",
            "--no_wandb",
            "--obj_eval",
        ],
        check=True,
    )
