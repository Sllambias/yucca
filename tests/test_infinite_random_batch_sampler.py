def test_inifite_random_sampler():
    import numpy as np
    from yucca.training.data_loading.samplers import RandomBatchSampler

    dataset = np.arange(100)
    batch_size = 10
    sampler = RandomBatchSampler(dataset, batch_size)

    # repeat test 5 times
    for i in range(5):
        sample = list(sampler)[0]
        assert len(sample) == batch_size
