def test_loads_env_var():
    from yucca.paths import get_yucca_raw_data

    assert get_yucca_raw_data() is not None
