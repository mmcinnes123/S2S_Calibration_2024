

from helpers_2DoF import get_J1_J2_from_opt

from os.path import join
import opensim as osim
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import pandas as pd
import qmt



# Function to set the template model default pose based on the target pose we're using for calibration
def set_default_model_pose(model_file, pose):

    if pose in ['Alt_self', 'Alt_asst']:
        EL_X_new = 90  # Specify the default elbow flexion angle in degrees
    elif pose in ['N_self', 'N_asst']:
        EL_X_new = 0    # Specify the default elbow flexion angle in degrees
    else:
        EL_X_new = None
        print('Pose name not specified correctly')
        quit()

    osim.Model.setDebugLevel(-2)  # Stop warnings about missing geometry vtp files
    model = osim.Model(model_file)
    model.getCoordinateSet().get('EL_x').setDefaultValue(EL_X_new * np.pi / 180)
    model.printToXML(model_file)

    print(f'\nIMU das3.osim default elbow angle has been updated to {EL_X_new} degrees.')


# Use OpenSim's built-in calibration method
def osim_calibrate_model(cal_oris_file_path, calibrated_model_dir, model_file, method_name):

    # Calibration settings
    sensor_to_opensim_rotations = osim.Vec3(0, 0, 0)
    baseIMUName = 'thorax_imu'
    baseIMUHeading = '-x'  # Which axis of the thorax IMU points in same direction as the model's thorax x-axis?

    from constants_IMC import calibration_settings_template_file

    # Instantiate an IMUPlacer object
    imuPlacer = osim.IMUPlacer(calibration_settings_template_file)

    # Set properties for the IMUPlacer
    imuPlacer.set_model_file(model_file)
    imuPlacer.set_orientation_file_for_calibration(cal_oris_file_path)
    imuPlacer.set_sensor_to_opensim_rotations(sensor_to_opensim_rotations)
    imuPlacer.set_base_imu_label(baseIMUName)
    imuPlacer.set_base_heading_axis(baseIMUHeading)

    # Update settings file
    imuPlacer.printToXML(calibrated_model_dir + "\\" + 'IMU_Calibration_Settings.xml')

    # Run the IMUPlacer
    visualize_calibration = False
    imuPlacer.run(visualize_calibration)

    # Get the model with the calibrated IMU
    model = imuPlacer.getCalibratedModel()

    # Print the calibrated model to file.
    model.printToXML(calibrated_model_dir + r'\Calibrated_model_' + method_name + '.osim')

    print('Written calibrated model to ', calibrated_model_dir)


# Function to apply METHOD_4d
def get_IMU_offsets_custom_method(subject_code, IMU_type, opt_trial_name, event_to_start, event_to_end):

    # Specify the pose to use to calibrate the thorax IMU
    pose_name = 'N_self'
    pose_trial_name = 'CP'
    opt_method = 'rot_noDelta'

    from constants_IMC import template_model_file, sample_rate
    baseIMUHeading = '-x'  # Which axis of the thorax IMU points in same direction as the model's thorax x-axis?

    # Get the dict with the timings for FE and PS events
    subject_event_dict = get_event_dict_from_file(subject_code)

    # Get the IMU orientation data at calibration pose time
    cal_oris_file_path_1 = get_cal_ori_file_path(subject_code, pose_trial_name, pose_name, IMU_type)
    thorax_IMU_ori1, humerus_IMU_ori1, radius_IMU_ori1 = read_sto_quaternion_file(cal_oris_file_path_1)

    # Get model body orientations in ground during default pose
    thorax_ori, humerus_ori, radius_ori = get_model_body_oris_during_default_pose(template_model_file, pose_name)

    # Get heading offset between IMU heading and model heading
    heading_offset = get_heading_offset(thorax_ori, thorax_IMU_ori1, baseIMUHeading, debug=False)

    # Apply the heading offset to the IMU orientations
    heading_offset_ori = R.from_euler('y', heading_offset)  # Create a heading offset scipy rotation
    thorax_IMU_ori_rotated1 = heading_offset_ori * thorax_IMU_ori1

    """ FINDING FE AND PS FROM OPTIMISATION RESULT """

    # Get the estimated FE and PS axes from the optimisation
    opt_FE_axis_in_humerus_IMU, opt_PS_axis_in_radius_IMU, opt_results = \
        get_J1_J2_from_opt(subject_code, IMU_type, opt_method, opt_trial_name,
                           subject_event_dict, event_to_start, event_to_end, sample_rate, debug=False)

    # Get the body-IMU offset for each body, based on the custom methods specified in cal_method_dict
    thorax_virtual_IMU = get_IMU_cal_POSE_BASED(thorax_IMU_ori_rotated1, thorax_ori)
    humerus_virtual_IMU = get_IMU_cal_hum_method_6(opt_FE_axis_in_humerus_IMU, debug=False)
    radius_virtual_IMU = get_IMU_cal_rad_method_3(opt_PS_axis_in_radius_IMU, debug=False)

    return thorax_virtual_IMU, humerus_virtual_IMU, radius_virtual_IMU



# A function which takes an uncalibrated model (with IMUs already associated with each body)
# and inputs an euler orientation offset which defines the virtual IMU offset relative to the bodies
def apply_cal_to_model(thorax_virtual_IMU, humerus_virtual_IMU, radius_virtual_IMU, model_file, results_dir, method_name):

    model = osim.Model(model_file)

    def set_IMU_transform(virtual_IMU_R, body_name, imu_name):

        # Define a new OpenSim rotation from the scipy rotation
        mat_from_scipy = virtual_IMU_R.as_matrix()
        rot = osim.Rotation(osim.Mat33(mat_from_scipy[0,0], mat_from_scipy[0,1], mat_from_scipy[0,2],
                                       mat_from_scipy[1,0], mat_from_scipy[1,1], mat_from_scipy[1,2],
                                       mat_from_scipy[2,0], mat_from_scipy[2,1], mat_from_scipy[2,2]))

        # Apply the rotation to the IMU in the model
        IMU_frame = model.getBodySet().get(body_name).getComponent(imu_name)    # Get the exsisting phyiscal offset frame of the IMU
        trans_vec = IMU_frame.getOffsetTransform().T()    # Get the existing translational offset of the IMU frame
        transform = osim.Transform(rot, osim.Vec3(trans_vec))   # Create an opensim transform from the rotation and translation
        IMU_frame.setOffsetTransform(transform)  # Update the IMU frame transform

    set_IMU_transform(thorax_virtual_IMU, body_name='thorax', imu_name='thorax_imu')
    set_IMU_transform(humerus_virtual_IMU, body_name='humerus_r', imu_name='humerus_r_imu')
    set_IMU_transform(radius_virtual_IMU, body_name='radius_r', imu_name='radius_r_imu')

    model.setName("IMU_Calibrated_das")
    model.printToXML(results_dir + r'\Calibrated_model_' + method_name + '.osim')


def get_event_dict_from_file(subject_code):

    from constants_IMC import subject_event_files_dir
    event_file_name = subject_code + '_event_dict.txt'
    event_file = join(subject_event_files_dir, event_file_name)

    file_obj = open(event_file, 'r')
    event_dict_str = file_obj.read()
    file_obj.close()
    event_dict = eval(event_dict_str)

    return event_dict


# Get the file path for the sto file containing the IMU orientation data during the specified pose
def get_cal_ori_file_path(subject_code, trial_name, pose_name, IMU_type):
    from constants_IMC import sto_dir
    sto_path = join(sto_dir, subject_code, trial_name, IMU_type, pose_name + '_quats.sto')
    return sto_path


# Read the IMU orientation data from an sto file, extract first row, and return a scipy R
def read_sto_quaternion_file(IMU_orientations_file):

    # Read sto file
    with open(IMU_orientations_file, 'r') as file:
        df = pd.read_csv(file, header=5, sep="\t")

    df = df.iloc[0]

    # Create scipy orientations from the dataframe
    thorax_IMU_ori_np = np.fromstring(df.loc['thorax_imu'], sep=",")
    thorax_IMU_ori = R.from_quat([thorax_IMU_ori_np[1], thorax_IMU_ori_np[2], thorax_IMU_ori_np[3], thorax_IMU_ori_np[0]])
    humerus_IMU_ori_np = np.fromstring(df.loc['humerus_r_imu'], sep=",")
    humerus_IMU_ori = R.from_quat([humerus_IMU_ori_np[1], humerus_IMU_ori_np[2], humerus_IMU_ori_np[3], humerus_IMU_ori_np[0]])
    radius_IMU_ori_np = np.fromstring(df.loc['radius_r_imu'], sep=",")
    radius_IMU_ori = R.from_quat([radius_IMU_ori_np[1], radius_IMU_ori_np[2], radius_IMU_ori_np[3], radius_IMU_ori_np[0]])

    return thorax_IMU_ori, humerus_IMU_ori, radius_IMU_ori


# Get the orientation of each model body, relative to the ground frame, during the default pose
def get_model_body_oris_during_default_pose(model_file, pose_name):

    """ Make sure model is in correct pose """

    # Set the template model pose
    set_default_model_pose(model_file, pose_name)

    """ Get model body orientations in ground during default pose """

    # Create the model and the bodies
    model = osim.Model(model_file)
    thorax = model.getBodySet().get('thorax')
    humerus = model.getBodySet().get('humerus_r')
    radius = model.getBodySet().get('radius_r')

    # Unlock any locked coordinates to allow model to realise any position
    for i in range(model.getCoordinateSet().getSize()):
        model.getCoordinateSet().get(i).set_locked(False)

    # Create a state based on the model's default state
    default_state = model.initSystem()

    # Get the orientation of each body in the given state
    thorax_ori = get_scipyR_of_body_in_ground(thorax, default_state)
    humerus_ori = get_scipyR_of_body_in_ground(humerus, default_state)
    radius_ori = get_scipyR_of_body_in_ground(radius, default_state)

    return thorax_ori, humerus_ori, radius_ori

def get_scipyR_of_body_in_ground(body, state):
    Rot = body.getTransformInGround(state).R()
    quat = Rot.convertRotationToQuaternion()
    scipyR = R.from_quat([quat.get(1), quat.get(2), quat.get(3), quat.get(0)])
    return scipyR


# Get the heading offset between the thorax IMU heading and the heading of the model in its default state
def get_heading_offset(base_body_ori, base_IMU_ori, base_IMU_axis_label, debug):

    # Calculate the heading offset
    if base_IMU_axis_label == 'x':
        base_IMU_axis = (base_IMU_ori.as_matrix()[:, 0])  # The x-axis of the IMU in ground frame
    elif base_IMU_axis_label == '-x':
        base_IMU_axis = (-base_IMU_ori.as_matrix()[:, 0])  # The x-axis of the IMU in ground frame
    else:
        print("Error: Need to add code if axis is different from 'x'")
        quit()

    base_body_axis = base_body_ori.as_matrix()[:, 0]  # The x-axis of the base body in ground frame

    # Calculate the angle between IMU axis and base segment axis
    heading_offset = np.arccos(np.dot(base_body_axis, base_IMU_axis) /
                               (np.linalg.norm(base_body_axis) * np.linalg.norm(base_IMU_axis)))

    # Update the sign of the heading offset
    if base_IMU_axis[2] < 0:  # Calculate the sign of the rotation (if the z-component of IMU x-axis is negative, rotation is negative)
        heading_offset = -heading_offset

    if debug:
        print("Heading offset is: " + str(round(heading_offset * 180 / np.pi, 2)))
        print("(i.e. IMU heading is rotated " + str(round(-heading_offset * 180 / np.pi, 2))
              + " degrees around the vertical axis, relative to the model's default heading.")

    return heading_offset


# This method uses the default pose of the model to calculate an initial orientation offset between body and IMU.
# It replicates the built-in OpenSim calibration
def get_IMU_cal_POSE_BASED(IMU_ori, body_ori):

    # Find the body frame, expressed in IMU frame
    body_inIMU = IMU_ori.inv() * body_ori

    # Express the virtual IMU frame in the model's body frame
    virtual_IMU = body_inIMU.inv()

    return virtual_IMU


# Function to define the humerus IMU offset based on an estimated elbow flexion axis, supplemented by manual-calibration
# Note: this is based on the fixed carry angle defined in the model
def get_IMU_cal_hum_method_6(FE_axis_in_humerus_IMU, debug):

    """ GET MODEL EF AXIS IN HUMERUS FRAME """

    # Based on how the hu joint is defined in the model, the XYZ euler ori offset of the parent frame,
    # relative to humerus frame is:
    hu_parent_rel2_hum_R = R.from_euler('XYZ', [0, 0, 0.32318], degrees=False)

    # Based on how the hu joint is defined in the model, relative to the hu joint parent frame,
    # the vector of hu rotation axis (EL_x) is:
    FE_axis_rel2_hu_parent = [0.969, -0.247, 0]

    # Get the vector of hu rotation axis, relative to the humerus frame
    FE_axis_in_humerus = hu_parent_rel2_hum_R.apply(FE_axis_rel2_hu_parent)

    """ GET THE POSE-BASED IMU OFFSET TO CONSTRAIN RESULTS """

    # Get the body-IMU offset for each body, based on the pose-based method (mirroring OpenSims built-in calibration)
    manual_virtual_IMU = get_IMU_cal_MANUAL('Humerus')

    # Get the individual axes of the pose-based virtual IMU frame
    y_comp_of_manual_based_offset = manual_virtual_IMU.as_matrix()[:, 1]

    """ FIND OPTIMAL VIRTUAL IMU OFFSET BASED ON THE INPUTS """

    # We are trying to find a rotational offset between two frames, A - the model's humerus, and B - the humerus IMU
    # The scipy align_vectors() function finds a rotational offset between two frames which optimally aligns two sets of
    # vectors defined in those frames: a, and b.
    # The largest weight is given to the first pair of vectors, because we want to strictly enforce that the estimated
    # elbow flexion axis is aligned with the model elbow flexion axis.
    # The other pairs of vectors are included to constrain the undefined DoF which would be present if we only used the
    # elbow flexion axis vectors. These pairs try to align the humerus IMU frame with the initial estimate of virtual
    # IMU frame from the pose-based calibration

    # Specify the first pairs of vectors which should be aligned, with the highest weighting
    a1 = FE_axis_in_humerus
    b1 = FE_axis_in_humerus_IMU
    w1 = 100

    # Specify the other pairs of vectors, using the initial guess at the IMU offset based on pose
    a2 = y_comp_of_manual_based_offset   # i.e. the axis of the pose-based virtual IMU frame
    b2 = [0, 1, 0]    # i.e. the x, y, z axis of the IMU frame
    w2 = 1         # These are weighted much lower because we want to prioritise the flexion axis estimation

    # Compile the arrays
    a = [a1, a2]
    b = [b1, b2]
    w = [w1, w2]

    # Alternative function
    virtual_IMU_quat, Wahba_debug = qmt.quatFromVectorObservations(b, a, weights=w, debug=True, plot=debug)

    # Convert the virtual IMU offset to a scipy R
    virtual_IMU = R.from_quat([virtual_IMU_quat[1], virtual_IMU_quat[2], virtual_IMU_quat[3], virtual_IMU_quat[0]])

    if debug:
        print("The estimated FE axis in the humerus IMU frame is:", FE_axis_in_humerus_IMU)
        print("The model's EF axis in the humerus frame is: ", FE_axis_in_humerus)
        print("The initial estimate of virtual IMU offset from manual calibration is: \n", manual_virtual_IMU.as_matrix())
        print("The optimal virtual IMU offset is: \n", virtual_IMU.as_matrix())
        print("The optimal virtual IMU offset is: \n", virtual_IMU.as_quat())

        """ PLOT THE OPTIMISATION """

        # Function to plot an arrow in 3D
        def plot_arrow(ax, start, direction, color, linewidth, length, label):
            ax.plot([start[0], start[0] + direction[0]*length],
                    [start[1], start[1] + direction[1]*length],
                    [start[2], start[2] + direction[2]*length], color=color, linewidth=linewidth)
            if label != None:
                ax.text(start[0] + direction[0]*length*1.1, start[1] + direction[1]*length*1.1, start[2] + direction[2]*length*1.1, label,
                        color=color, fontsize=12)

        # Create a new figure
        fig = plt.figure()

        # Define the unit vectors for the x, y, and z axes
        x_axis = np.array([1, 0, 0])
        y_axis = np.array([0, 1, 0])
        z_axis = np.array([0, 0, 1])

        # Plot all the vectors in frame A
        ax1 = fig.add_subplot(131, projection='3d')
        origin = np.array([1, 1, 1])
        plot_arrow(ax1, origin, x_axis, 'black', linewidth=3, length=1.3, label='X')
        plot_arrow(ax1, origin, y_axis, 'black', linewidth=3, length=1.3, label='Y')
        plot_arrow(ax1, origin, z_axis, 'black', linewidth=3, length=1.3, label='Z')
        plot_arrow(ax1, origin, a1, 'purple', linewidth=3, length=0.8, label='FE_ref')
        plot_arrow(ax1, origin, a2, 'blue', linewidth=2, length=0.8, label='y_man')

        # Plot all the vectors in frame B
        ax2 = fig.add_subplot(132, projection='3d')
        origin = np.array([1, 1, 1])
        # Plot the x, y, and z axes as arrows with custom width
        plot_arrow(ax2, origin, x_axis, 'black', linewidth=3, length=1.3, label='X')
        plot_arrow(ax2, origin, y_axis, 'black', linewidth=3, length=1.3, label='Y')
        plot_arrow(ax2, origin, z_axis, 'black', linewidth=3, length=1.3, label='Z')
        plot_arrow(ax2, origin, b1, 'purple', linewidth=1, length=1.1, label='FE_opt')
        plot_arrow(ax2, origin, b2, 'blue', linewidth=1, length=1.1, label='y')

        # Apply the estimated rotation to the second set of vectors
        b1_rot = qmt.rotate(virtual_IMU_quat, b1)
        b2_rot = qmt.rotate(virtual_IMU_quat, b2)

        # Plot all the a vectors in frame A, and the rotated b vectors in frame A
        ax3 = fig.add_subplot(133, projection='3d')
        origin = np.array([1, 1, 1])
        # Plot the x, y, and z axes as arrows with custom width
        plot_arrow(ax3, origin, x_axis, 'black', linewidth=3, length=1.3, label='X')
        plot_arrow(ax3, origin, y_axis, 'black', linewidth=3, length=1.3, label='Y')
        plot_arrow(ax3, origin, z_axis, 'black', linewidth=3, length=1.3, label='Z')
        plot_arrow(ax3, origin, a1, 'purple', linewidth=3, length=0.8, label='')
        plot_arrow(ax3, origin, a2, 'blue', linewidth=2, length=0.8, label='')
        plot_arrow(ax3, origin, b1_rot, 'purple', linewidth=1, length=1.1, label='FE')
        plot_arrow(ax3, origin, b2_rot, 'blue', linewidth=1, length=1.1, label='y')

        axes = [ax1, ax2, ax3]

        for ax in axes:
            # Set the limits
            ax.set_xlim([0, 2])
            ax.set_ylim([0, 2])
            ax.set_zlim([0, 2])
            ax.invert_zaxis()
            ax.invert_xaxis()
            # Adjust the view angle so that the y-axis points upwards
            ax.view_init(elev=0, azim=180)
            # Remove ticks and tick labels
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

        # Show the plot
        plt.show()

    return virtual_IMU


# This function calculates the IMU offset required which is equivalent to relying on 'manual alignment'
# The only reason we need to apply an offset (and not just have 0 offset) is because the IMU axis names 'xyz' don't
# match the names of the body axes, so are only rotated in multiples of 90degrees
def get_IMU_cal_MANUAL(which_body):

    if which_body == "Thorax":
        virtual_IMU = R.from_euler('XYZ', [0, 180, 0], degrees=True)

    elif which_body == "Humerus":
        virtual_IMU = R.from_euler('XYZ', [180, 90, 0], degrees=True)

    elif which_body == "Radius":
        virtual_IMU = R.from_euler('XYZ', [0, 0, 180], degrees=True)

    else:
        print("Which_body input wasn't 'Thorax', 'Humerus', or 'Radius'")

    return virtual_IMU


# Function to define the radius IMU offset based on an estimated pronation/supination axis,
# supplemented by manual calibration
def get_IMU_cal_rad_method_3(PS_axis_in_radius_IMU, debug):

    """ GET MODEL PS AXIS IN RADIUS FRAME """

    # PS_axis of the ur joint is defined relative to the parent/child frames, where the child frame = radius body frame
    PS_axis_in_radius = [0.182, 0.98227, -0.044946]

    """ GET THE POSE-BASED IMU OFFSET TO CONSTRAIN RESULTS """

    # Get the IMU offset defined by a manual alignment, to refine results of the optimisation estimation
    manual_virtual_IMU = get_IMU_cal_MANUAL('Radius')

    # Get the individual axes of the manual virtual IMU frame
    x_comp_of_manual_offset = manual_virtual_IMU.as_matrix()[:, 0]

    """ FIND OPTIMAL VIRTUAL IMU OFFSET BASED ON THE INPUTS """

    # We are trying to find a rotational offset between two frames, A - the model's radius, and B - the radius IMU
    # The scipy align_vectors() function finds a rotational offset between two frames which optimally aligns two sets of
    # vectors defined in those frames: a, and b.
    # The largest weight is given to the first pair of vectors, because we want to strictly enforce that the estimated
    # PS axis is aligned with the model PS axis.
    # The other pairs of vectors are included to constrain the undefined DoF which would be present if we only used the
    # PS axis vectors. These pairs try to align the radius IMU frame with the initial estimate of virtual
    # IMU frame from the manual calibration

    # Specify the first pair of vectors which should be aligned, with the highest weighting
    a1 = PS_axis_in_radius
    b1 = PS_axis_in_radius_IMU
    w1 = 10000

    # Specify the other pair of vectors, using the initial guess at the IMU offset based on manual alignment
    a2 = x_comp_of_manual_offset   # i.e. the axis of the manual virtual IMU frame
    b2 = [1, 0, 0]  # i.e. the x, y, z axis of the IMU frame
    w2 = 1  # These are weighted much lower because we want to prioritise the flexion axis estimation

    # Compile the arrays
    a = [a1, a2]
    b = [b1, b2]
    w = [w1, w2]

    # Alternative function
    virtual_IMU_quat, Wahba_debug = qmt.quatFromVectorObservations(b, a, weights=w, debug=True, plot=debug)

    # Convert the virtual IMU offset to a scipy R
    virtual_IMU = R.from_quat([virtual_IMU_quat[1], virtual_IMU_quat[2], virtual_IMU_quat[3], virtual_IMU_quat[0]])

    if debug:
        print("The estimated PS axis in the radius IMU frame is:", PS_axis_in_radius_IMU)
        print("The model's PS axis in the radius frame is: ", PS_axis_in_radius)
        print("The initial estimate of virtual IMU offset from manual calibration is: \n", manual_virtual_IMU.as_matrix())
        print("The optimal virtual IMU offset is: \n", virtual_IMU.as_matrix())
        print("The optimal virtual IMU offset is: \n", virtual_IMU.as_quat())

        """ PLOT THE OPTIMISATION """

        # Function to plot an arrow in 3D
        def plot_arrow(ax, start, direction, color, linewidth, length, label):
            ax.plot([start[0], start[0] + direction[0]*length],
                    [start[1], start[1] + direction[1]*length],
                    [start[2], start[2] + direction[2]*length], color=color, linewidth=linewidth)
            if label != None:
                ax.text(start[0] + direction[0]*length*1.1, start[1] + direction[1]*length*1.1, start[2] + direction[2]*length*1.1, label,
                        color=color, fontsize=12)

        # Create a new figure
        fig = plt.figure()

        # Define the unit vectors for the x, y, and z axes
        x_axis = np.array([1, 0, 0])
        y_axis = np.array([0, 1, 0])
        z_axis = np.array([0, 0, 1])

        # Plot all the vectors in frame A
        ax1 = fig.add_subplot(131, projection='3d')
        origin = np.array([1, 1, 1])
        plot_arrow(ax1, origin, x_axis, 'black', linewidth=3, length=1.3, label='X')
        plot_arrow(ax1, origin, y_axis, 'black', linewidth=3, length=1.3, label='Y')
        plot_arrow(ax1, origin, z_axis, 'black', linewidth=3, length=1.3, label='Z')
        plot_arrow(ax1, origin, a1, 'purple', linewidth=3, length=0.8, label='FE_ref')
        plot_arrow(ax1, origin, a2, 'blue', linewidth=2, length=0.8, label='y_man')

        # Plot all the vectors in frame B
        ax2 = fig.add_subplot(132, projection='3d')
        origin = np.array([1, 1, 1])
        # Plot the x, y, and z axes as arrows with custom width
        plot_arrow(ax2, origin, x_axis, 'black', linewidth=3, length=1.3, label='X')
        plot_arrow(ax2, origin, y_axis, 'black', linewidth=3, length=1.3, label='Y')
        plot_arrow(ax2, origin, z_axis, 'black', linewidth=3, length=1.3, label='Z')
        plot_arrow(ax2, origin, b1, 'purple', linewidth=1, length=1.1, label='FE_opt')
        plot_arrow(ax2, origin, b2, 'blue', linewidth=1, length=1.1, label='y')

        # Apply the estimated rotation to the second set of vectors
        b1_rot = qmt.rotate(virtual_IMU_quat, b1)
        b2_rot = qmt.rotate(virtual_IMU_quat, b2)

        # Plot all the a vectors in frame A, and the rotated b vectors in frame A
        ax3 = fig.add_subplot(133, projection='3d')
        origin = np.array([1, 1, 1])
        # Plot the x, y, and z axes as arrows with custom width
        plot_arrow(ax3, origin, x_axis, 'black', linewidth=3, length=1.3, label='X')
        plot_arrow(ax3, origin, y_axis, 'black', linewidth=3, length=1.3, label='Y')
        plot_arrow(ax3, origin, z_axis, 'black', linewidth=3, length=1.3, label='Z')
        plot_arrow(ax3, origin, a1, 'purple', linewidth=3, length=0.8, label='')
        plot_arrow(ax3, origin, a2, 'blue', linewidth=2, length=0.8, label='')
        plot_arrow(ax3, origin, b1_rot, 'purple', linewidth=1, length=1.1, label='FE')
        plot_arrow(ax3, origin, b2_rot, 'blue', linewidth=1, length=1.1, label='y')

        axes = [ax1, ax2, ax3]

        for ax in axes:
            # Set the limits
            ax.set_xlim([0, 2])
            ax.set_ylim([0, 2])
            ax.set_zlim([0, 2])
            ax.invert_zaxis()
            ax.invert_xaxis()
            # Adjust the view angle so that the y-axis points upwards
            ax.view_init(elev=0, azim=180)
            # Remove ticks and tick labels
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

        # Show the plot
        plt.show()

    return virtual_IMU