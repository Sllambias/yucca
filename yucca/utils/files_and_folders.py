import pkgutil
import importlib
import fileinput
import re
import shutil
import os
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    subfiles,
    subdirs,
)


def replace_in_file(file_path, pattern_replacement):
    with fileinput.input(file_path, inplace=True) as file:
        for line in file:
            for pattern, replacement in pattern_replacement.items():
                line = line.replace(pattern, replacement)
            print(line, end="")


def rename_file_or_dir(file: str, patterns: dict):
    # Patterns is a dict of key, value pairs where keys are the words to replace and values are
    # what to substitute them by. E.g. if patterns = {"foo": "bar"}
    # then the sentence "foo bar" --> "bar bar"
    newfile = file
    for k, v in patterns.items():
        newfile = re.sub(k, v, newfile)
    if os.path.isdir(file):
        if newfile != file:
            shutil.move(file, newfile)
    elif os.path.isfile(file):
        os.rename(file, newfile)
    return newfile


def recursive_rename(folder, patterns_in_file={}, patterns_in_name={}, file_ext=".py"):
    """
    Takes a top folder and recursively looks through all subfolders and files.
    For all file contents it will replace patterns_in_file keys with the corresponding values.
    For all file names it will replace the patterns_in_name keys with the corresponding values.

    If patterns_in_file = {"llama": "alpaca", "coffee": "TEA"}
    and patterns_in_name = {"foo": "bar", "Foo": "Bar"}
    and we take the file:

    MyPythonFooScript.py
    ---- (with the following lines of code) ----
    llama = 42
    coffee = 123

    something_else = llama + coffee
    ----

    then we will end up with

    MyPythonBarScript.py
    ----
    alpaca = 42
    TEA = 123

    something_else = alpaca + TEA
    """
    dirs = subdirs(folder)
    files = subfiles(folder)
    for file in files:
        if len(patterns_in_file) > 0:
            if os.path.splitext(file)[-1] == file_ext:
                replace_in_file(
                    file,
                    patterns_in_file,
                )
        rename_file_or_dir(file, patterns_in_name)
    for direc in dirs:
        direc = rename_file_or_dir(direc, patterns_in_name)
        recursive_rename(direc, patterns_in_file, patterns_in_name)


def recursive_find_python_class(folder: list, class_name: str, current_module: str):
    """
    Stolen from nnUNet model_restore.py.
    Folder = starting path, e.g. join(yucca.__path__[0], 'preprocessing')
    Trainer_name = e.g. YuccaPreprocessor3D
    Current_module = starting module e.g. 'yucca.preprocessing'

    The function is nested to allow it to exit after the recursive part is done. This lets us to
    check if we have successfully found what we looked for IN the function.
    """

    def _recursive_find_python_class(folder: list, class_name: str, current_module: str):
        tr = None
        for _, modname, ispkg in pkgutil.iter_modules(folder):
            if not ispkg:
                m = importlib.import_module(current_module + "." + modname)
                if hasattr(m, class_name):
                    tr = getattr(m, class_name)
                    break

        if tr is None:
            for _, modname, ispkg in pkgutil.iter_modules(folder):
                if ispkg:
                    next_current_module = current_module + "." + modname
                    tr = _recursive_find_python_class(
                        [join(folder[0], modname)],
                        class_name,
                        current_module=next_current_module,
                    )
                if tr is not None:
                    break
        return tr

    found_python_class = _recursive_find_python_class(folder, class_name, current_module)
    assert (
        found_python_class is not None
    ), f"Did not find any python class called {class_name}. Make sure there's no typos (and that the class actually exists)"
    return found_python_class


def recursive_find_realpath(path):
    """
    This might produce undesirable results if run on a slurm/batch management user, that does not
    the same permissions as you.
    """
    non_linked_dirs = []
    while path:
        if os.path.islink(path):
            path = os.path.realpath(path)
        path, part = os.path.split(path)
        non_linked_dirs.append(part)
        if path == os.path.sep:
            non_linked_dirs.append(path)
            path = False
    return os.path.join(*non_linked_dirs[::-1])
