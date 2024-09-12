def single_case_collate(x):
    # Takes the standard batched list and just returns the single item in the list.
    return x[0]
