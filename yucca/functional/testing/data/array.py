import numpy as np


def verify_labels_are_equal(expected_labels: np.ndarray, actual_labels: np.ndarray, id=""):
    if expected_labels.dtype == actual_labels.dtype:
        if np.all(np.isin(actual_labels, expected_labels)):
            return True
        else:
            print(f"Unexpected labels found for {id} \n" f"expected: {expected_labels} \n" f"found: {actual_labels}")
            return False
    else:
        print(
            "make sure reference and target is the same dtype before comparing the labels. \n"
            f"reference is: {expected_labels.dtype} and target is: {actual_labels.dtype}"
        )
        return False


def verify_array_shape_is_equal(reference: np.ndarray, target: np.ndarray, id=""):
    if np.all(reference.shape == target.shape):
        return True
    else:
        print(f"Sizes do not match for {id}" f"Image is: {reference.shape} while the label is {target.shape}")
        return False


def verify_shape_is_equal(reference, target, id=""):
    if np.all(reference == target):
        return True
    else:
        print(f"Sizes do not match for {id}" f"Image is: {reference.shape} while the label is {target.shape}")
        return False
