from yucca.modules.data.augmentation.transforms.YuccaTransform import YuccaTransform
from yucca.functional.transforms import skeleton


class Skeleton(YuccaTransform):
    def __init__(self, skeleton=False, label_key="label", do_tube=False):
        self.label_key = label_key
        self.skeleton = skeleton
        self.do_tube = do_tube

    @staticmethod
    def get_params():
        # No parameters to retrieve
        pass

    def __skeletonize__(self, label):
        label_skeleton = label.copy()
        for b in range(label.shape[0]):
            for c in range(label.shape[1]):
                label_skeleton[b, c] = skeleton(label[b, c])
        return label_skeleton[0]

    def __call__(self, packed_data_dict=None, **unpacked_data_dict):
        data_dict = packed_data_dict if packed_data_dict else unpacked_data_dict
        if self.skeleton:
            data_dict["skel"] = self.__skeletonize__(data_dict[self.label_key])
        return data_dict
