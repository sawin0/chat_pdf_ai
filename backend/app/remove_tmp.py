import os
import shutil


def remove_tmp_files() -> None:
    tmp_dir = "./tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    for entry in os.scandir(tmp_dir):
        if entry.is_dir(follow_symlinks=False):
            shutil.rmtree(entry.path)
        else:
            os.remove(entry.path)
