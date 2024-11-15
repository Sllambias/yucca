import logging

logging.getLogger().setLevel(logging.INFO)


# Tell PyInstaller where to find hook-yucca.py
def _pyinstaller_hooks_dir():
    from pathlib import Path

    return [str(Path(__file__).with_name("_pyinstaller").resolve())]
