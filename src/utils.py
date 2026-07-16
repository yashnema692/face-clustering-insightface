import os
import shutil


def create_folder(path):

    os.makedirs(path, exist_ok=True)


def copy_image(source, destination):

    shutil.copy2(source, destination)