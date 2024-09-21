# This script reads the raw (x, y, z) marker position data, converting the file from .txt to .trc,
# ready to read in OpenSim
# Input is a MM report .txt file
# Output is a .trc file

from helpers_OMC import MM_2_trc
from constants_OMC import sample_rate, OMC_dir, raw_data_dir, trc_dir

from os.path import join
from os import makedirs
from tkinter.filedialog import askdirectory, askopenfilename


def run_preprocess(subject_code, trial_name, test):

    # Input settings manually if running single test
    if test:
        input_file_path = str(askopenfilename(title=' Choose the TMM txt report file containing marker position data ... '))
        output_file_dir = str(askdirectory(title=' Choose the folder where the .trc file will be saved ... '))
        output_file_name = input("Name of output file: (include .trc)")
        output_file_path = join(output_file_dir, output_file_name)

    else:
        # Define some directories and file names
        input_file_name = subject_code + '_' + trial_name + r' - Report1 - Marker_Traj.txt'
        output_file_name = subject_code + '_' + trial_name + '_marker_pos.trc'
        input_file_path = join(raw_data_dir, input_file_name)
        output_file_path = join(trc_dir, output_file_name)

        # Create folders if they don't already exist
        makedirs(OMC_dir, exist_ok=True)
        makedirs(trc_dir, exist_ok=True)

    # Convert MM .txt file to .trc file
    MM_2_trc(input_file_path, sample_rate, output_file_path)


if __name__ == '__main__':
    run_preprocess(subject_code=None, trial_name=None, test=True)


