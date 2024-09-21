# This script runs the scale tool in OpenSim, which simultaneously scales the model bodies and moves the model markers
# Input is the template model file and a trc file with marker data at a static pose time
# Output is a scaled model (without markers moved), a scaled model with markers moved, and a static pose .mot showing
# results of the single IK step

from helpers_OMC import run_osim_scale_tool

from os.path import join
from os import makedirs
import opensim as osim
from tkinter.filedialog import askdirectory, askopenfilename


""" SETTINGS """


def run_scale_model(subject_code, static_time_dict, test):

    from constants_OMC import scale_settings_template_file, OMC_dir

    # Input settings manually if running single test
    if test:
        OMC_template_model = str(askopenfilename(title=' Choose the .osim model file which is the template for scaling ... '))
        trc_file_name_for_scaling = str(askopenfilename(title=' Choose the .trc file used for the scale IK step (usually CP)... '))
        subject_dir = str(askdirectory(title=' Choose the folder where the scaled model file will be saved ... '))
        time_in_trc_for_scaling = input('Enter the pose time from the trc file used for the IK step (s):')
    else:

        # Define some file paths
        subject_dir = join(OMC_dir, subject_code)
        trc_file_name_for_scaling = subject_code + r'_CP_marker_pos.trc'  # The movement data used to scale the OMC model
        makedirs(subject_dir, exist_ok=True)

        from constants_OMC import OMC_template_model

        if subject_code in static_time_dict.keys():
            time_in_trc_for_scaling = static_time_dict[subject_code]
        else:
            time_in_trc_for_scaling = None
            print('QUIT MESSAGE: You need to add the pose time to the static_time_dict for subject ', subject_code)
            quit()

    # Create a log file
    osim.Logger.addFileSink(subject_dir + r'\calibration.log')

    # Scale the model
    run_osim_scale_tool(scale_settings_template_file, OMC_template_model,
                        time_in_trc_for_scaling, trc_file_name_for_scaling, subject_code)


""" TEST """

if __name__ == '__main__':
    run_scale_model(subject_code=None, static_time_dict=None, test=True)
