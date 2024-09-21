
from create_trc_OMC import run_createtrc
from scale_model_OMC import run_scale_model
from inverse_kinematics_OMC import run_OMC_IK


""" PREPROCESS """
# Converts .txt file from TMM to a .trc file in correct format for OpenSim

createtrc = True
if createtrc:

    # Choose which subjects to process
    subject_code_list = [f'P{str(i).zfill(3)}' for i in range(3, 21)]

    # Choose which movement trials to process
    trial_name_list = ['CP', 'JA_Slow', 'JA_Fast', 'ROM', 'ADL']

    # Iterate through the collection of subjects and movement trials
    for subject_code in subject_code_list:

        for trial_name in trial_name_list:

            run_createtrc(subject_code, trial_name, test=False)


""" SCALE MODEL """

scale_model = True
if scale_model:

    # Choose which subjects to process
    subject_code_list = ['P001']
    # subject_code_list = [f'P{str(i).zfill(3)}' for i in range(1, 21)] # Or run the function for all subjects

    # A dict defining the time to use for OMC calibration
    # (these times define the moment in the CP trial at pose: Alt_asst)
    static_time_dict = {'P001': 30, 'P002': 29, 'P003': 23, 'P004': 21, 'P005': 36,
                        'P006': 38, 'P007': 44, 'P008': 38, 'P009': 41, 'P010': 28,
                        'P011': 38, 'P012': 40, 'P013': 20, 'P014': 34, 'P015': 32,
                        'P016': 40, 'P017': 25, 'P018': 32, 'P019': 21, 'P020': 26}

    for subject_code in subject_code_list:

        run_scale_model(subject_code, static_time_dict, test=False)


""" RUN IK """
# Calculate joint kinematics using the calibrated model and marker data from a certain trial

IK = True
if IK:

    # Choose which subjects to process
    subject_code_list = ['P001']
    # subject_code_list = [f'P{str(i).zfill(3)}' for i in range(1, 21)]   # Or run the function for all subjects

    # Choose which movement trials to process
    trial_name_list = ['CP', 'JA_Slow', 'ADL']
    # trial_name_list = ['CP', 'JA_Slow', 'JA_Fast', 'ROM', 'ADL']  # Or run the function for all trials

    # Iterate through the collection of subjects and movement types
    for subject_code in subject_code_list:

        for trial_name in trial_name_list:

            run_OMC_IK(subject_code, trial_name, run_analysis=True, test=False)
