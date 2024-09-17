def test_loads_env_var():
    from yucca.paths import get_raw_data_path

    assert get_raw_data_path() is not None
