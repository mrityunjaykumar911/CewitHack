import pypianoroll
import os

from tqdm import tqdm


def get_all_file_from_directory(path):
    all_files_with_full_path = []
    for path, subdirs, files in os.walk(path):
        for name in files:
            xx = os.path.join(path, name)
            all_files_with_full_path.append((name, xx))
    return all_files_with_full_path


all_files = get_all_file_from_directory("../data/lpd/")
out_dir_name = "../data/lakhdataset_midi/"


for sign, each in tqdm(all_files):
    filep = pypianoroll.load(filepath=each)
    filep.write("{path}{sign}.mid".format(path=out_dir_name,sign=sign))
