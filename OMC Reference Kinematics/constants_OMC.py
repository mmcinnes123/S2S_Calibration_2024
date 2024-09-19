# This script contains a list of constant settings used throughout the repo

from os.path import join

parent_dir = r'C:\Users\r03mm22\Documents\Protocol_Testing\S2S_Calibration_2024'
OMC_dir = join(parent_dir, 'OMC Reference Kinematics')
raw_data_dir = join(parent_dir, 'Raw Data')
trc_dir = join(OMC_dir, 'TRC_Position_Data')

sample_rate = 100
OMC_template_model = 'das3.osim'

# Settings files
scale_settings_template_file = 'OMC_Scale_Settings.xml'  # See run_scale_model() for more settings
IK_settings_template_file = 'OMC_IK_Settings.xml'  # See run_OMC_IK() for more settings
analyze_settings_template_file = 'Analyze_Settings.xml'
