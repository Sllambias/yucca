import numpy as np


def verify_labels_are_equal(expected_labels: np.ndarray, actual_labels: np.ndarray, id=""):
    assert expected_labels.dtype == actual_labels.dtype, (
        "make sure reference and target is the same dtype before comparing the labels. \n"
        f"reference is: {expected_labels.dtype} and target is: {actual_labels.dtype}"
    )
    assert np.all(np.isin(actual_labels, expected_labels)), (
        f"Unexpected labels found for {id} \n" f"expected: {expected_labels} \n" f"found: {actual_labels}"
    )


def verify_array_shape_is_equal(reference: np.ndarray, target: np.ndarray, id=""):
    assert np.all(reference.shape == target.shape), (
        f"Sizes do not match for {id}" f"Image is: {reference.shape} while the label is {target.shape}"
    )


def verify_shape_is_equal(reference, target, id=""):
    assert np.all(reference == target), (
        f"Sizes do not match for {id}" f"Image is: {reference.shape} while the label is {target.shape}"
    )
