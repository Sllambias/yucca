import re
from yucca.preprocessing.YuccaPreprocessor import YuccaPreprocessor
from yucca.paths import yucca_preprocessed_data, yucca_raw_data
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    subfiles,
)


class UnsupervisedPreprocessor(YuccaPreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # set up for self-/unsupervised
        self.classification = False
        self.label_exists = False
        self.preprocess_label = False

    def initialize_paths(self):
        # Have to overwrite how we get the subject_ids as there's no labelsTr to get them from.
        # Therefore we use the imagesTr folder and remove the modality suffix.
        self.target_dir = join(yucca_preprocessed_data, self.task, self.plans["plans_name"] + "__" + self.name)
        self.input_dir = join(yucca_raw_data, self.task)
        self.imagepaths = subfiles(join(self.input_dir, "imagesTr"), suffix=self.image_extension)

        subject_ids = subfiles(join(self.input_dir, "imagesTr"), suffix=self.image_extension, join=False)
        self.subject_ids = [re.sub(r"_\d+\.", ".", subject) for subject in subject_ids]
