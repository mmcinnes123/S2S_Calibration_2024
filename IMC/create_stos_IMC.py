# This script takes the sensor orientation data from the TMM .txt file and copy and pastes the data into an .sto file,
# ready for use in OpenSim. It also adds appropriate column headings so that the IMU data is named correctly, e.g.
# IMU1 = thorax_IMU, IMU2 = humerus_r_IMU, IMU3 = radius_r_IMU. It also creates .sto files which contain one time
# instant of orientation data at different pose times, based on times which have been chosen manually and written in the
# SubjectEventFiles

from constants_IMC import subject_event_files_dir, sto_dir, raw_data_dir
from helpers_createstos import \
    get_trial_pose_time_dict_from_file, \
    read_data_frame_from_file, \
    write_data_to_sto, \
    write_pose_data_to_sto

from os import makedirs
from os.path import join
from tkinter.filedialog import askopenfilename, askdirectory


def run_create_stos_IMC(subject_code, test):

    if test:

        raw_data_file_path = str(askopenfilename(title=' Choose the input .txt file which you want to convert to .sto ... '))
        sto_file_dir = str(askdirectory(title=' Choose the folder where the .sto file will be saved ... '))
        # Read data from TMM .txt report
        IMU1_df, IMU2_df, IMU3_df = read_data_frame_from_file(raw_data_file_path)
        # Write all data to sto
        write_data_to_sto(IMU1_df, IMU2_df, IMU3_df, sto_file_dir, file_tag='All_quats')

    else:

        # Get the dict containing the times for each event in each trial
        trial_name_dict = get_trial_pose_time_dict_from_file(subject_event_files_dir, subject_code)

        # Run the function for all the trials we have data for
        for trial_name in trial_name_dict:

            # For both the 'real' and 'perfect' IMUs
            IMU_type_dict = {'Real': ' - Report2 - IMU_Quats.txt', 'Perfect': ' - Report3 - Cluster_Quats.txt'}

            for IMU_key in IMU_type_dict:

                print('Creating preprocessed data (.stos) for subject: ', subject_code,
                      'for trial: ', trial_name,
                      'for IMU type: ', IMU_key)

                # Get the times specific to this trial
                cal_pose_time_dict = trial_name_dict[trial_name]

                # Create the .sto file folder to save new files
                sto_file_dir = join(sto_dir, subject_code, trial_name, IMU_key)
                makedirs(sto_file_dir, exist_ok=True)

                # Specify the input file
                raw_data_file_path = join(raw_data_dir, subject_code + '_' + trial_name + IMU_type_dict[IMU_key])

                # Read data from TMM .txt report
                IMU1_df, IMU2_df, IMU3_df = read_data_frame_from_file(raw_data_file_path)

                # Write all data to sto
                write_data_to_sto(IMU1_df, IMU2_df, IMU3_df, sto_file_dir, file_tag='All_quats')

                # Write single rows of data to separate .stos, using time samples defined manually in SubjectEventFile
                write_pose_data_to_sto(IMU1_df, IMU2_df, IMU3_df, sto_file_dir, cal_pose_time_dict)



""" TEST """

if __name__ == '__main__':
    run_create_stos_IMC(subject_code=None, test=True)
