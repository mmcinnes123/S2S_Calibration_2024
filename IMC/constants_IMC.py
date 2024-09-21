# A list of constants used throughout the whole repository

from os.path import join

# Directories
parent_dir = r'C:\Users\r03mm22\Documents\Protocol_Testing\S2S_Calibration_2024'
raw_data_dir = join(parent_dir, 'Raw Data')
subject_event_files_dir = join(parent_dir, 'SubjectEventFiles')
IMC_dir = join(parent_dir, 'IMC')
sto_dir = join(IMC_dir, 'STO Data')

# Settings files
template_model_file = join(IMC_dir, 'IMC_model_das3.osim')
APDM_template_file = join(IMC_dir, 'APDM_template_4S.csv')
APDM_settings_file = join(IMC_dir, 'APDMDataConverter_Settings.xml')
calibration_settings_template_file = join(IMC_dir, 'IMU_Calibration_Settings.xml')
IMU_IK_settings_file = join(IMC_dir, 'IMU_IK_Settings.xml')
analyze_settings_template_file = join(IMC_dir, 'Analyze_Settings.xml')

# Plot settings
prominence = 20     # Set the minimum prominence which qualifys a peak in teh find_peaks() function

sample_rate = 100
