The repository holds the data and code used to support worked published here: link


Change the parent dict in both `constants_OMC.py` and `constants_IMC.py`. 
Note that 'pre-processing' has been done - i.e. creating .stos and .trcs
Say what all poses are: N_asst, Alt_asst, self, Alt2 etc.

#### OMC Reference Kinematics 
This section of the repo is for calculating joint kinematics from the recorded marker position data, using an OpenSim 
model and OpenSim inverse kinematics.

There are three basic steps:
(These functions can be run separately, or everything can be batch run from `main_OMC.py`.)
1. Create a .trc file - `create_trc_OMC.py` - simply copy and pastes the raw data which is in a .txt file, into a .trc 
file with all the right headings and formatting so that OpenSim can read it without complaining.
2. Create a scaled model - `scale_model_OMC.py` - for each individual, using the OpenSim scale model tool, adjusts the 
dimensions of the bodies in the generic OpenSim model (OMC_model_das3.osim), based on measured distances between the 
markers (look at the OMC_Scale_Settings.xml for details), and at the same time, adjusts the position of markers in the
model to better match their position during measurement.  
3. Run the inverse kinematics - `inverse_kinematics_OMC.py` - using time-series marker data, and the scaled model, to 
run OpenSim's inverse kinematics tool, which calculates the pose of the model on a frame-by-frame basis. Note, the output 
of this step is the .mot _joint kinematics_ file, but also a .sto 'analysis' file, which gives us access to the time-series
_body kinematics_ which later allows us to calculate humero-thoracic angles, not gleno-humeral.


#### IMC 
This section of the repo is for handling the IMU data, using OpenSim functions and custom S2S calibrations.
