"""
Adaptation from BSL6D (https://gitlab.mpcdf.mpg.de/bsl6d/bsl6d)

Download and extract graphics for the GEMPICX documentation.

https://datashare.mpcdf.mpg.de/s/9DGFfLmoi1SPCEe

If the directory for the graphics already exists it is 
deleted and newly created.
If a link to the directory exists, the data is overwritten but
not skipped.
"""
import os
import shutil
import subprocess
import zipfile


def graphics_dir_name():
    return "./graphics"


def link_to_graphics_exists():
    return (os.path.islink(graphics_dir_name()))


def download_graphics_directory():
    subprocess.run(r'curl -k -o graphics.zip "https://datashare.mpcdf.mpg.de/s/9DGFfLmoi1SPCEe/download"',
                   shell=True, check=True)


def extract_graphics_directory():
    if (os.path.exists(graphics_dir_name())):
        shutil.rmtree(graphics_dir_name())
    # Extract and rename folder. Datashare folder can have an arbitrary name
    with zipfile.ZipFile("graphics.zip", "r") as f:
        f.extractall()
        folderName = f.namelist()[0]
    os.rename(folderName, "graphics")


def main(sourceDir):
    print("Downloading graphics from MPCDF Datashare\n")
    currentDir = os.getcwd()
    os.chdir(sourceDir)
    if (not os.path.exists('./_static/')):
        os.makedirs("./_static/")
    os.chdir("./_static")
    if (not link_to_graphics_exists()):
        try:
            download_graphics_directory()
            extract_graphics_directory()
        except Exception as e:
            print("\n[WARNING] Could not download or extract graphics folder. Will continue build process with missing graphics.")
    else:
        print("\n[WARNING] Download and extraction of graphics directory is skipped"
                      "because a link '" + graphics_dir_name() + "' already exists!")
    os.chdir(currentDir)


if __name__ == '__main__':
    # Execute in location of Python sript
    main(os.path.dirname(os.path.realpath(__file__)))
