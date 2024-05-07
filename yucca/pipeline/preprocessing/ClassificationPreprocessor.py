import torch
from yucca.pipeline.preprocessing.YuccaPreprocessor import YuccaPreprocessor


class ClassificationPreprocessor(YuccaPreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # set up for classification
        self.classification = True
        self.label_exists = True
        self.preprocess_label = False

    def reverse_preprocessing(self, images: torch.Tensor, image_properties: dict):
        """
        Expected shape of images are:
        (b, c, x)
        """
        image_properties["save_format"] = "txt"
        return images.cpu().numpy(), image_properties
