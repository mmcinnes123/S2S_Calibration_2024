# This script employs different S2S calibration methods

from constants_IMC import template_model_file
from helpers_calibration_IMC import \
    set_default_model_pose, \
    osim_calibrate_model, \
    get_IMU_offsets_custom_method, \
    apply_cal_to_model

from os.path import join
from os import makedirs


def run_osim_calibration_IMC(subject_code, pose_name, IMU_type):

    from constants_IMC import sto_dir
    from constants_IMC import IMC_dir

    method_name = 'OSIM_' + pose_name

    # Use a pose from the CP trial (poses from other trials could be used)
    trial_name = 'CP'

    # Get the path to the orientations file specific to the chosen pose
    cal_oris_file_path = join(sto_dir, subject_code, trial_name, IMU_type, pose_name + '_quats.sto')

    # Set the template model pose
    set_default_model_pose(template_model_file, pose_name)

    # Get/make the directory to save the calibrated model
    calibrated_model_dir = join(IMC_dir, subject_code, method_name, IMU_type, 'Calibrated Model')
    makedirs(calibrated_model_dir, exist_ok=True)

    # Run the opensim calibration
    osim_calibrate_model(cal_oris_file_path, calibrated_model_dir, template_model_file, method_name)



def run_custom_calibration_IMC(subject_code, IMU_type, movement_name):

    if movement_name == 'ISO_1rep':
        opt_trial_name = 'JA_Slow'
        event_to_start = 'FE5_start'
        event_to_end = 'PS2_start'

    elif movement_name == 'ISO_5reps':
        opt_trial_name = 'JA_Slow'
        event_to_start = 'FE_start'
        event_to_end = 'PS_end'

    elif movement_name == 'ADL_both':
        opt_trial_name = 'ADL'
        if subject_code == 'P008':
            event_to_start = 'drink1_start'
            event_to_end = 'kettle1_end'
        else:
            event_to_start = 'kettle1_start'
            event_to_end = 'drink1_end'

    elif movement_name == 'ADL_drink':
        opt_trial_name = 'ADL'
        event_to_start = 'drink1_start'
        event_to_end = 'drink1_end'

    elif movement_name == 'ADL_kettle':
        opt_trial_name = 'ADL'
        event_to_start = 'kettle1_start'
        event_to_end = 'kettle1_end'

    else:
        opt_trial_name = None
        event_to_start = None
        event_to_end = None
        print('Method name not written properly')
        quit()

    thorax_virtual_IMU, humerus_virtual_IMU, radius_virtual_IMU = \
        get_IMU_offsets_custom_method(subject_code, IMU_type, opt_trial_name, event_to_start, event_to_end)

    # Get/make the directory to save the calibrated model
    method_name = 'CUSTOM_' + movement_name
    from constants_IMC import IMC_dir
    calibrated_model_dir = join(IMC_dir, subject_code, method_name, IMU_type, 'Calibrated Model')
    makedirs(calibrated_model_dir, exist_ok=True)

    # Create the calibrated model, applying the calculated offsets to the default model
    apply_cal_to_model(thorax_virtual_IMU, humerus_virtual_IMU, radius_virtual_IMU, template_model_file,
                       calibrated_model_dir, method_name)
