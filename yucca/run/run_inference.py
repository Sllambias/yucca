import argparse
import yucca
import warnings
from yucca.utils.files_and_folders import recursive_find_python_class, maybe_get_task_from_task_id
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, \
    maybe_mkdir_p, isdir
from yucca.paths import yucca_raw_data, yucca_results, yucca_models
from yucca.evaluation.YuccaEvaluator import YuccaEvaluator
from yucca.training.trainers.YuccaTrainer import YuccaTrainer
from yuccatemp.yucca.utils.merge_softmax import merge_softmax_from_folders



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", help="Name of the source task i.e. what the model is trained on. "
                        "Should be of format: TaskXXX_MYTASK", required=True)
    parser.add_argument("-t", help="Name of the target task i.e. the data to be predicted. "
                        "Should be of format: TaskXXX_MYTASK", required=True)
    parser.add_argument("-f", help="Select the fold that was used to train the model desired for inference. "
                        "Defaults to looking for a model trained on fold 0.", default="0")
    parser.add_argument("-m", help="Model Architecture. Defaults to UNet.", default="UNet")
    parser.add_argument("-d", help="2D, 25D or 3D model. Defaults to 3D.", default='3D')
    parser.add_argument("-tr", help="Full name of Trainer Class. \n"
                        "e.g. 'YuccaTrainer_DCE' or 'YuccaTrainerV2'. Defaults to YuccaTrainerV2.", default='YuccaTrainerV2')
    parser.add_argument("-pl", help="Plan ID. Defaults to YuccaPlannerV2", default="YuccaPlannerV2")
    parser.add_argument("-chk", help="Checkpoint to use for inference. Defaults to checkpoint_best.", default="checkpoint_best")
    parser.add_argument("--ensemble", help="Used to initialize data preprocessing for ensemble/2.5D training", default=False,
                        action='store_true')
    parser.add_argument("--do_tta", help="Used to enable test-time augmentations (mirroring)", default=False,
                        action='store_true')
    parser.add_argument("--not_strict", default=False, action='store_true', required=False,
                        help="Strict determines if all expected modalities must be present, "
                        "with the appropriate suffixes (e.g. '_000.nii.gz'). "
                        "Only touch if you know what you're doing.")
    parser.add_argument("--save_softmax", default=False, action='store_true', required=False,
                        help="Save softmax outputs. Required for softmax fusion.")
    parser.add_argument("--overwrite", default=False, action='store_true', required=False,
                        help="Overwrite existing predictions")
    parser.add_argument("--no_eval", help="Disable evaluation and creation of metrics file (result.json)",
                        default=False, action='store_true', required=False)
    parser.add_argument("--predict_train", default=False, action='store_true', required=False,
                        help="Predict on the training set. Useful for debugging.")
    # parser.add_argument("--threads", help="number of threads/processes", default=2)

    args = parser.parse_args()

    source_task = maybe_get_task_from_task_id(args.s)
    target_task = maybe_get_task_from_task_id(args.t)
    trainer_name = args.tr
    model = args.m
    dimensions = args.d
    folds = args.f
    plan_id = args.pl
    checkpoint = args.chk
    ensemble = args.ensemble
    do_tta = args.do_tta
    not_strict = args.not_strict
    save_softmax = args.save_softmax
    overwrite = args.overwrite
    predict_train = args.predict_train
    no_eval = args.no_eval
    # threads = args.threads

    warnings.simplefilter("ignore", ResourceWarning)
    folders_with_softmax = []
    if ensemble:
        print("Running ensemble inference on the default ensemble plans \n"
              "Save_softmax set to True.")
        plans = [plan_id+"X", plan_id+"Y", plan_id+"Z"]
        save_softmax = True
    else:
        plans = [plan_id]

    for plan in plans:
        modelfile = join(yucca_models, source_task, model, dimensions,
                        trainer_name + '__' + plan, folds, checkpoint + '.model')
        assert isfile(modelfile), "Can't find .model file with trained model weights. "\
            f"Should be located at: {modelfile}"
        print(f"######################################################################## \n"
              f"{'Using model: ':25} {modelfile}")

        metafile = modelfile+'.json'
        assert isfile(metafile), "Can't find .json file with model metadata. "\
            f"Should be located at: {metafile}"
        metafile = load_json(metafile)

        """
        We find the trainer using the name stored in the modelfile and NOT the "trainer_name" argument.
        E.g. if the "--lr 1e-4" flag is used with the base YuccaTrainer the models will be saved as 
        YuccaTrainer_1e4 even though YuccaTrainer_1e4 might not exist. Therefore we refer to the 
        modelfile for the actual Trainer used.
        """
        trainer_class = metafile["trainer_class"]
        trainer = recursive_find_python_class(folder=[join(yucca.__path__[0], 'training')],
                                            class_name=trainer_class, current_module='yucca.training')

        assert trainer, f"searching for {trainer_class} "\
            f"but found: {trainer}"
        assert issubclass(trainer, YuccaTrainer), "Trainer is not a subclass of YuccaTrainer."

        print(f"{'Using trainer: ':25} {trainer}")
        trainer = trainer(model, dimensions, task=source_task, folds=folds, plan_id=plan)

        # Setting up input paths and output paths
        inpath = join(yucca_raw_data, target_task, 'imagesTs')
        ground_truth = join(yucca_raw_data, target_task, 'labelsTs')

        outpath = join(yucca_results, target_task, source_task, model+dimensions,
                    trainer_name + '__' + plan,
                    'fold_' + folds + '_' + checkpoint)

        if predict_train:
            inpath = join(yucca_raw_data, target_task, 'imagesTr')
            ground_truth = join(yucca_raw_data, target_task, 'labelsTr')
            outpath += 'Tr'

        maybe_mkdir_p(outpath)

        trainer.load_checkpoint(modelfile)
        trainer.predict_folder(inpath, outpath, not_strict=not_strict, save_softmax=save_softmax,
                               overwrite=overwrite, do_tta=do_tta)

        folders_with_softmax.append(outpath)

        if isdir(ground_truth) and not no_eval:
            evaluator = YuccaEvaluator(trainer.classes, folder_with_predictions=outpath,
                                      folder_with_ground_truth=ground_truth)
            evaluator.run()

    if ensemble:
        ensemble_outpath = join(yucca_results, target_task, source_task, model+dimensions,
                                trainer_name + '__' + plan_id+'_Ensemble',
                                'fold_' + folds + '_' + checkpoint)
        merge_softmax_from_folders(folders_with_softmax, ensemble_outpath)

        if isdir(ground_truth) and not no_eval:
            evaluator = YuccaEvaluator(trainer.classes, folder_with_predictions=ensemble_outpath,
                                      folder_with_ground_truth=ground_truth)
            evaluator.run()


if __name__ == '__main__':
    main()
