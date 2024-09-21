

from create_stos_IMC import run_create_stos_IMC
from calibration_IMC import run_osim_calibration_IMC, run_custom_calibration_IMC

create_stos = False
if create_stos:

    # Choose which subjects to process
    subject_code_list = ['P001', 'P002']
    # subject_code_list = [f'P{str(i).zfill(3)}' for i in range(1, 21)]   # Or run the function for all subjects

    for subject_code in subject_code_list:

        run_create_stos_IMC(subject_code, test=False)


run_osim_calibration = False
if run_osim_calibration:

    # Choose which subjects to process
    subject_code_list = ['P001', 'P002']
    # subject_code_list = [f'P{str(i).zfill(3)}' for i in range(1, 21)]   # Or run the function for all subjects

    IMU_type = 'Real'       # Options: Real, Perfect
    pose_name = 'N_self'    # Options: N_self, N_asst, Alt_self, Alt_asst

    for subject_code in subject_code_list:

        run_osim_calibration_IMC(subject_code, pose_name, IMU_type)


run_custom_calibration = False
if run_custom_calibration:

    # Choose which subjects to process
    subject_code_list = ['P002']
    # subject_code_list = [f'P{str(i).zfill(3)}' for i in range(1, 21)]   # Or run the function for all subjects

    IMU_type = 'Real'       # Options: Real, Perfect

    # Choose the trial and action from which to extract sensor data for the optimisation
    movement_name = 'ISO_1rep'      # Options: ISO_1rep, ISO_5reps, ADL_both, ADL_drink, ADL_kettle

    for subject_code in subject_code_list:

        run_custom_calibration_IMC(subject_code, IMU_type, movement_name)
