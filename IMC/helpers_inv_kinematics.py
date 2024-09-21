
from os.path import join
import opensim as osim
import os
import numpy as np

def get_event_dict_from_file(subject_code):
    event_files_folder = r'C:\Users\r03mm22\Documents\Protocol_Testing\2024 Data Collection\SubjectEventFiles'
    event_file_name = subject_code + '_event_dict.txt'
    event_file = join(event_files_folder, event_file_name)

    file_obj = open(event_file, 'r')
    event_dict_str = file_obj.read()
    file_obj.close()
    event_dict = eval(event_dict_str)

    return event_dict




def run_osim_IMU_IK(IMU_IK_settings_file, calibrated_model_file, orientations_file,
               sensor_to_opensim_rotations, results_directory, start_at_pose_bool, trim_bool,
               start_time, end_time, IK_output_file_name, visualize_tracking):

    osim.Model.setDebugLevel(0)  # Stop warnings about missing geometry vtp files (set to -2)

    # Instantiate an InverseKinematicsTool
    imuIK = osim.IMUInverseKinematicsTool(IMU_IK_settings_file)

    # Set tool properties
    imuIK.set_model_file(calibrated_model_file)
    imuIK.set_orientations_file(orientations_file)
    imuIK.set_sensor_to_opensim_rotations(sensor_to_opensim_rotations)
    imuIK.set_results_directory(results_directory)

    if start_at_pose_bool:
        imuIK.set_time_range(0, start_time)
    if trim_bool:
        imuIK.set_time_range(0, start_time)
        imuIK.set_time_range(1, end_time)

    imuIK.setOutputMotionFileName(IK_output_file_name)

    # Set IMU weights
    thorax_imu_weight = osim.OrientationWeight('thorax_imu', 1.0)
    humerus_imu_weight = osim.OrientationWeight('humerus_r_imu', 0.1)
    radius_imu_weight = osim.OrientationWeight('radius_r_imu', 1.0)
    imuIK.upd_orientation_weights().cloneAndAppend(thorax_imu_weight)
    imuIK.upd_orientation_weights().cloneAndAppend(humerus_imu_weight)
    imuIK.upd_orientation_weights().cloneAndAppend(radius_imu_weight)

    # Run IK
    print('Running IMU IK...')
    imuIK.run(visualize_tracking)
    print('IMU IK run finished.')

    # Update the settings .xml file
    imuIK.printToXML(results_directory + "\\" + 'IMU_IK_Settings.xml')



def run_analysis(subject_code, trial_name, calibration_name, IMU_type):

    """ SETTINGS """

    from constants_IMC import analyze_settings_template_file, IMC_dir

    # Define some file paths
    sub_cal_dir = join(IMC_dir, subject_code, calibration_name, IMU_type)
    IK_results_dir = join(sub_cal_dir, 'IK_results')
    calibrated_model_file = 'Calibrated_model_' + calibration_name + '.osim'
    calibrated_model_path = join(sub_cal_dir, 'Calibrated Model', calibrated_model_file)
    coord_file_for_analysis = join(IK_results_dir, "IMU_IK_results.mot")

    # Create opensim logger file
    osim.Logger.removeFileSink()
    osim.Logger.addFileSink(IK_results_dir + r'\Analysis.log')

    # Set start and end time by checking length of data
    coords_table = osim.TimeSeriesTable(coord_file_for_analysis)
    start_time = np.round(coords_table.getIndependentColumn()[0], 2)
    end_time = np.round(coords_table.getIndependentColumn()[-1], 2)

    """ MAIN """

    print(f'Running analysis for {subject_code}, {calibration_name}, {trial_name}, {IMU_type}.')

    run_analyze_tool(analyze_settings_template_file, IK_results_dir, calibrated_model_path, coord_file_for_analysis, start_time, end_time)


def run_analyze_tool(analyze_settings_template_file, results_dir, model_file_path, mot_file_path, start_time, end_time):

    model = osim.Model(model_file_path)
    analyze_Tool = osim.AnalyzeTool(analyze_settings_template_file)
    analyze_Tool.updAnalysisSet().cloneAndAppend(osim.BodyKinematics())
    analyze_Tool.setModel(model)
    analyze_Tool.setName("analyze")
    analyze_Tool.setCoordinatesFileName(mot_file_path)
    analyze_Tool.setStartTime(start_time)
    analyze_Tool.setFinalTime(end_time)
    analyze_Tool.setResultsDir(results_dir)
    print('Running Analyze Tool...')
    analyze_Tool.run()
    print('Analyze Tool run finished.')