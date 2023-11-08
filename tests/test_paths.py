def test_loads_env_var():
    from yucca.paths import yucca_preprocessed

    assert yucca_preprocessed is not None
