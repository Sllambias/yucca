def test_loads_env_var():
    from yucca.paths import yucca_raw_data

    assert yucca_raw_data is not None
