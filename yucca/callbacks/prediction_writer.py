from lightning.pytorch.callbacks import BasePredictionWriter
from yucca.functional.utils.saving import save_prediction_from_logits, save_multilabel_prediction_from_logits
from batchgenerators.utilities.file_and_folder_operations import join


class WritePredictionFromLogits(BasePredictionWriter):
    def __init__(self, output_dir, multilabel: bool = False, save_softmax: bool = False, write_interval: str = "batch"):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.multilabel = multilabel
        self.save_softmax = save_softmax

    def write_on_batch_end(self, _trainer, _pl_module, data_dict, _batch_indices, _batch, _batch_idx, _dataloader_idx):
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        logits, properties, case_id = (
            data_dict["logits"],
            data_dict["properties"],
            data_dict["case_id"],
        )
        if self.multilabel:
            save_multilabel_prediction_from_logits(
                logits,
                join(self.output_dir, case_id),
                properties=properties,
            )
        else:
            save_prediction_from_logits(
                logits,
                join(self.output_dir, case_id),
                properties=properties,
                save_softmax=self.save_softmax,
            )
        del data_dict
