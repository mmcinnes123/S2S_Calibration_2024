
import opensim as osim
import math
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R


# Function to convert .txt file from TMM to a .trc file
def MM_2_trc(input_file_name, sample_rate, output_file_name):

    # Create new empty variables and dataframe
    marker_pos = []
    labelset = []
    data_out = {'DataRate': sample_rate,
            'CameraRate': sample_rate,
            'NumFrames': 1,
            'NumMarkers': 1,
            'Units': 'm',
            'OrigDataRate': sample_rate,
            'OrigDataStartFrame': 1,
            'OrigNumFrames': 1,
            'Labels': [],
            'Data': [],
            'Timestamps': []
                }

    # Read in MM .txt file
    print(f"Reading in data from {input_file_name}...")
    dataset = pd.read_csv(input_file_name, delimiter="\t")

    # Extract all the data and append to Data_out dataframe
    for index in range(dataset.shape[0]):
        test = []
        for counter in range(1,dataset.shape[1]-1):
            test.append(dataset.iloc[index][counter])
        marker_pos.append(test)

    for row in marker_pos:
        data_out['Data'].append(row)

    # Extract and edit marker label names
    for col in dataset.columns:
        labelset.append(str(col))
    # remove first and last item from list as they are not labels
    labelset = labelset[1:-1]
    # keep every third marker name (there are three for X, Y, Z)
    labelset = labelset[::3]
    # replace any spaces with underscores to be allowed in opensim
    converter = lambda x: x.replace(' ', '.')
    labelset = list(map(converter, labelset))
    # remove _X from every label
    converter_2 = lambda x: x.replace('_X', '')
    labelset = list(map(converter_2, labelset))

    # Create metadata for data_out dataframe
    num_markers = (len(data_out['Data'][0])) / 3
    num_frames = len(data_out['Data'])

    # Create new time column
    timestamps = list(np.arange(0, num_frames / sample_rate, 1 / sample_rate))

    # Add info into dataframe
    data_out['Timestamps'] = timestamps
    data_out['DataRate'] = sample_rate
    data_out['CameraRate'] = sample_rate
    data_out['OrigDataRate'] = sample_rate
    data_out['NumMarkers'] = num_markers
    data_out['NumFrames'] = num_frames
    data_out['OrigNumFrames'] = num_frames
    data_out['Labels'] = labelset

    # Check if the characters in strings are in ASCII, U+0-U+7F.
    def isascii(s):
        return len(s) == len(s.encode())

    # Write the data_out into new TRC file
    fullname = output_file_name
    outputfile = open(fullname, "w")
    writeTRC(data_out, outputfile)
    outputfile.close()

    print(f"Written data to {output_file_name}.")


# Function to write the data_out into a TRC file
def writeTRC(data, file):

    # Write header
    file.write("PathFileType\t4\t(X/Y/Z)\toutput.trc\n")
    file.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
    file.write("%d\t%d\t%d\t%d\t%s\t%d\t%d\t%d\n" % (data["DataRate"], data["CameraRate"], data["NumFrames"],
                                                     data["NumMarkers"], data["Units"], data["OrigDataRate"],
                                                     data["OrigDataStartFrame"], data["OrigNumFrames"]))

    # Write labels
    file.write("Frame#\tTime\t")
    for i, label in enumerate(data["Labels"]):
        if i != 0:
            file.write("\t")
        # file.write("\t\t%s" % (label))
        file.write("%s\t\t" % (label))
    file.write("\n")
    file.write("\t")
    for i in range(len(data["Labels"]*3)):
        file.write("\t%c%d" % (chr(ord('X')+(i%3)), math.ceil((i+1)/3)))
    file.write("\n")

    # Write data_out
    for i in range(len(data["Data"])):
        file.write("%d\t%f" % (i, data["Timestamps"][i]))
        for l in range(len(data["Data"][0])):
            file.write("\t%f" % data["Data"][i][l])

        file.write("\n")