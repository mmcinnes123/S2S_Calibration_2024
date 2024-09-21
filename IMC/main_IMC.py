

from create_stos_IMC import run_create_stos_IMC
from calibration_IMC import run_osim_calibration_IMC, run_custom_calibration_IMC
from inv_kinematics_IMC import run_inv_k_IMC
from helpers_inv_kinematics import run_analysis

create_stos = False
if create_stos:

    # Choose which subjects to process (named P001 to P020)
    subject_code_list = ['P001', 'P002']
    # subject_code_list = [f'P{str(i).zfill(3)}' for i in range(1, 21)]   # Or run the function for all subjects

    for subject_code in subject_code_list:
        run_create_stos_IMC(subject_code, test=False)


run_osim_calibration = False
if run_osim_calibration:

    # Choose which subjects to process (named P001 to P020)
    subject_code_list = ['P001', 'P002']
    # subject_code_list = [f'P{str(i).zfill(3)}' for i in range(1, 21)]   # Or run the function for all subjects

    # Chose which type of IMU data to use
    IMU_type = 'Real'       # Options: Real, Perfect

    # Choose which pose to use for the calibration
    pose_name = 'N_self'    # Options: N_self, N_asst, Alt_self, Alt_asst

    for subject_code in subject_code_list:
        run_osim_calibration_IMC(subject_code, pose_name, IMU_type)


run_custom_calibration = False
if run_custom_calibration:

    # Choose which subjects to process (named P001 to P020)
    subject_code_list = ['P002']
    # subject_code_list = [f'P{str(i).zfill(3)}' for i in range(1, 21)]   # Or run the function for all subjects

    # Chose which type of IMU data to use
    IMU_type = 'Real'       # Options: Real, Perfect

    # Choose the trial and action from which to extract sensor data for the optimisation
    movement_name = 'ISO_1rep'      # Options: ISO_1rep, ISO_5reps, ADL_both, ADL_drink, ADL_kettle

    for subject_code in subject_code_list:
        run_custom_calibration_IMC(subject_code, IMU_type, movement_name)


run_inv_k = True
if run_inv_k:

    # Choose which subjects to process (named P001 to P020)
    subject_code_list = ['P001']
    # subject_code_list = [f'P{str(i).zfill(3)}' for i in range(1, 21)]   # Or run the function for all subjects

    # Chose which type of IMU data to use
    IMU_type = 'Real'       # Options: Real, Perfect

    # State the calibration method name to specify which calibrated model is used
    calibration_name = 'CUSTOM_ISO_1rep'     # Must match name of existing folder. e.g. CUSTOM_ISO_1rep or OSIM_N_self

    # Choose which movement trial to calculate the inverse kinematics for
    trial_name = 'JA_Slow'

    # Set this to true to make the IK automatically start for the last calibration pose to save time
    IK_start_at_pose_bool = True

    # Set to true if you want to run an OpenSim 'analysis' which will calculate and save body orientation data (takes longer)
    get_body_ori_data = True

    for subject_code in subject_code_list:
        run_inv_k_IMC(subject_code, trial_name, calibration_name, IK_start_at_pose_bool, IMU_type)

        if get_body_ori_data:
            run_analysis(subject_code, trial_name, calibration_name, IMU_type)


# TODO: Add Compare Module Somewhere