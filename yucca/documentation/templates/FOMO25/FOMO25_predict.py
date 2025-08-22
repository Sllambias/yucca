from yucca.modules.callbacks.prediction_writer import WriteSinglePredictionFromLogits


def predict(input_path, output_path):
    from config import (
        manager,
        modelfile,
        experiment,
        model_name,
        model_dimensions,
        source_task,
        planner,
    )
    from yucca.modules.data.datasets.alternative_datasets.SingleFileDataset import (
        SingleFileTestDataset,
    )

    output_path = output_path.replace(".nii.gz", "")
    manager = manager(
        ckpt_path=modelfile,
        enable_logging=False,
        experiment=experiment,
        num_workers=0,
        model_name=model_name,
        model_dimensions=model_dimensions,
        task=source_task,
        planner=planner,
    )

    manager.batch_size = 1
    manager.test_dataset_class = SingleFileTestDataset
    manager.initialize(
        stage="predict",
        disable_tta=True,
        disable_inference_preprocessing=False,
        overwrite_predictions=True,
        pred_data_dir=input_path,
        prediction_output_dir=output_path,
        save_softmax=False,
    )

    manager.trainer.callbacks = [
        WriteSinglePredictionFromLogits(
            output_path=output_path,
            multilabel=False,
            save_softmax=False,
            write_interval="batch",
        )
    ]

    manager.trainer.predict(
        model=manager.model_module,
        dataloaders=manager.data_module,
        ckpt_path=modelfile,
        return_predictions=False,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputpath",
        help="The path to a SINGLE file",
        default="",
    )
    parser.add_argument(
        "--outputpath",
        default="",
    )
    args = parser.parse_args()

    predict(args.inputpath, args.outputpath)
