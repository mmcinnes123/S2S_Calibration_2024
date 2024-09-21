# This script runs IMU IK with the OpenSense API
# Input is .sto files create in 1_preprocess.py, one with calibration pose, one with movements of interest
# Calibrates an .osim model by assigning IMUs to segments
# Outputs .mot IK results

from helpers_inv_kinematics import get_event_dict_from_file, run_osim_IMU_IK

import opensim as osim
import os
from os.path import join


def run_inv_k_IMC(subject_code, trial_name, calibration_name, start_at_pose_bool, IMU_type):

    IK_trim_bool = False
    IK_start_time = 7
    IK_end_time = 10

    from constants_IMC import IMU_IK_settings_file, IMC_dir, sto_dir

    """ SETTINGS """

    # IMU IK Settings
    IK_output_file_name = "IMU_IK_results.mot"
    visualize_tracking = False
    sensor_to_opensim_rotations = osim.Vec3(0, 0, 0)

    # Define some file paths
    sub_cal_dir = join(IMC_dir, subject_code, calibration_name, IMU_type)
    calibrated_model_file = 'Calibrated_model_' + calibration_name + '.osim'
    calibrated_model_path = join(sub_cal_dir, 'Calibrated Model', calibrated_model_file)
    orientations_file_path = join(sto_dir, subject_code, trial_name, IMU_type, 'All_quats.sto')
    IK_results_dir = join(sub_cal_dir, 'IK_results')
    os.makedirs(IK_results_dir, exist_ok=True)

    # Create opensim logger file
    osim.Logger.removeFileSink()
    osim.Logger.addFileSink(IK_results_dir + r'\IMU_IK.log')

    """ MAIN """

    # Check that the calibrated model has been created
    if os.path.exists(calibrated_model_path) == False:
        print(f"You haven't created the calibrated model for subject {subject_code}, IMU_type: {IMU_type}, calibration_name: {calibration_name} yet")
        print("Quitting.")
        quit()

    if start_at_pose_bool:
        # Get the time at which the alt pose is performed, to start the IK from there
        subject_event_dict = get_event_dict_from_file(subject_code)
        pose_time = subject_event_dict[trial_name]['Alt_self']
        IK_start_time = pose_time

    # Run the IMU IK based on settings inputs above
    print(f'Running IMU IK for {subject_code}, {calibration_name}, {trial_name}, {IMU_type}.')
    run_osim_IMU_IK(IMU_IK_settings_file, calibrated_model_path, orientations_file_path, sensor_to_opensim_rotations,
                    IK_results_dir, start_at_pose_bool, IK_trim_bool, IK_start_time, IK_end_time, IK_output_file_name, visualize_tracking)