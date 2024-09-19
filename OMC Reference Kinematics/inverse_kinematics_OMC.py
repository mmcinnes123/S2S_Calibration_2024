# This script performs marker-based IK with OpenSim API
# Input is a .trc file and a scaled OpenSim model
# Output is an .mot file

from helpers_OMC import run_osim_OMC_IK_tool, find_marker_error, run_analyze_tool
from constants_OMC import IK_settings_template_file, sample_rate, analyze_settings_template_file, OMC_dir, trc_dir

from os import makedirs
from os.path import join
import opensim as osim
from tkinter.filedialog import askopenfilename, askdirectory


def run_OMC_IK(subject_code, trial_name, run_analysis, test):

    """ SETTINGS """

    # Input settings/files manually if running single test
    if test:
        results_dir = str(askdirectory(title=' Choose the folder where the IK results will be saved ... '))
        scaled_model = str(askopenfilename(title=' Choose the scaled .osim model to run the IK ... '))
        trc_file_path = str(askopenfilename(title=' Choose the .trc file to run the IK ... '))
        trim_bool_str = input('IK trim bool (True or False):')
        if trim_bool_str == 'True':
            trim_bool = True
            IK_start_time = input('Start time (s): ')
            IK_end_time = input('End time (s): ')
        else:
            trim_bool = False
            IK_start_time = None
            IK_end_time = None
    else:

        # Define some file paths
        trc_file_path = join(trc_dir, subject_code + '_' + trial_name + r'_marker_pos.trc')  # Define a path to the marker data
        subject_dir = join(OMC_dir, subject_code)
        trial_dir = join(subject_dir, trial_name)
        scaled_model = join(subject_dir, 'das3_scaled_and_placed.osim')
        results_dir = trial_dir

        # Make the IK results directory if it doesn't already exist
        makedirs(results_dir, exist_ok=True)

        # Add a new opensim.log
        osim.Logger.addFileSink(results_dir + r'\IK.log')

        # Don't trim the data if we're iterating through subjects
        trim_bool = False
        IK_start_time = None
        IK_end_time = None

    """ MAIN """

    # Run the IK
    run_osim_OMC_IK_tool(IK_settings_template_file, trim_bool, IK_start_time, IK_end_time, results_dir,
                         trc_file_path, scaled_model)

    # Log the marker error
    find_marker_error(results_dir)

    """ ANALYSIS """

    if run_analysis:

        # Specify where to get the IK results file
        coord_file_for_analysis = join(results_dir, 'OMC_IK_results.mot')
        osim.Logger.addFileSink(results_dir + r'\analysis.log')

        # Set end time by checking length of data
        if trim_bool == False:
            if trial_name == 'ADL':     # Only run for the first 60s of the ADL trial
                analysis_start_time = 0
                analysis_end_time = 60
            else:
                coords_table = osim.TimeSeriesTable(coord_file_for_analysis)
                n_rows = coords_table.getNumRows()
                analysis_start_time = 0
                analysis_end_time = n_rows / sample_rate
        else:
            analysis_start_time = IK_start_time
            analysis_end_time = IK_end_time

        # Run the analyze tool to output the BodyKinematics.sto
        run_analyze_tool(analyze_settings_template_file, results_dir, scaled_model,
                         coord_file_for_analysis, analysis_start_time, analysis_end_time)


""" TEST """

if __name__ == '__main__':
    run_OMC_IK(subject_code=None, trial_name=None, run_analysis=True, test=True)
