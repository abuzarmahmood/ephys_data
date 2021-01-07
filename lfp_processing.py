"""
Module for ephys_data to streamline lfp_extraction
Adapted from blech_clust/LFP_analysis/LFP_Processing_Final.py
"""
# ==============================
# Setup
# ==============================

#Import necessary tools
import numpy as np
import tables
import easygui
import os
import glob
import matplotlib.pyplot as plt
import re
import sys
from tqdm import tqdm, trange
import shutil
#Import specific functions in order to filter the data file
from scipy.signal import butter
from scipy.signal import filtfilt

# ==============================
# Define Functions 
# ==============================

def get_filtered_electrode(data, low_pass, high_pass, sampling_rate):
    el = 0.195*(data)
    m, n = butter(
            2, 
            [2.0*int(low_pass)/sampling_rate, 2.0*int(high_pass)/sampling_rate], 
            btype = 'bandpass')
    filt_el = filtfilt(m, n, el)
    return filt_el

# ==============================
# Collect user input needed for later processing 
# ==============================

# call by: extract_lfps(dir_name, **lfp_param_dict)
def extract_lfps(dir_name,
                freq_bounds,
                sampling_rate,
                taste_signal_choice,
                fin_sampling_rate,
                dig_in_list,
                trial_durations):

    if taste_signal_choice is 'Start':
            diff_val = 1
    elif taste_signal_choice is 'End':
            diff_val = -1

    # ==============================
    # Open HDF5 File 
    # ==============================

    #Look for the hdf5 file in the directory
    hdf5_path = glob.glob(os.path.join(dir_name,'**.h5'))[0]

    #Open the hdf5 file
    hf5 = tables.open_file(hdf5_path, 'r+')

    # ==============================
    # Select channels to read
    # ==============================

    # Create vector of electode numbers that have neurons on them 
    # (from unit_descriptor table). 
    # Some electrodes may record from more than one neuron 
    # (shown as repeated number in unit_descriptor);
    # Remove these duplicates within array

    electrodegroup = np.unique(hf5.root.unit_descriptor[:]['electrode_number'])

    ## List all appropriate dat files
    Raw_Electrodefiles = np.sort(glob.glob(os.path.join(dir_name, '*amp*dat*')))
    Raw_Electrodefiles = Raw_Electrodefiles[electrodegroup]

    # ==============================
    # Extract Raw Data 
    # ==============================

    #Check if LFP data is already within file and remove node if so. 
    #Create new raw LFP group within H5 file. 
    if '/raw_LFP' in hf5:
        hf5.remove_node('/raw_LFP', recursive = True)
    hf5.create_group('/', 'raw_LFP')

    #Loop through each neuron-recording electrode (from .dat files), 
    #filter data, and create array in new LFP node

    # How many folds to downsample data by
    new_intersample_interval = sampling_rate/fin_sampling_rate

    # Pull out signal for each electrode, down_sample, 
    # bandpass filter and store in HDF5
    print('Extracting raw LFPs')
    for i in trange(len(Raw_Electrodefiles)):
        data = np.fromfile(Raw_Electrodefiles[i], dtype = np.dtype('int16'))
        data_down = np.mean(\
                data.reshape((-1, int(new_intersample_interval))), axis = -1)
        filt_el_down = get_filtered_electrode(data = data_down,
                                            low_pass = freq_bounds[0],
                                            high_pass = freq_bounds[1],
                                            sampling_rate = fin_sampling_rate)

        # Zero padding to 3 digits because code get screwy with sorting electrodes
        # if that isn't done
        hf5.create_array('/raw_LFP','electrode{:0>3}'.\
                format(electrodegroup[i]), filt_el_down)
        hf5.flush()
        del data, data_down, filt_el_down

    # Grab the names of the arrays containing digital inputs, 
    # and pull the data into a numpy array
    dig_in_nodes = hf5.list_nodes('/digital_in')
    dig_in = []
    dig_in_pathname = []
    for node in dig_in_nodes:
        dig_in_pathname.append(node._v_pathname)
        exec("dig_in.append(hf5.root.digital_in.%s[:])" \
                    % dig_in_pathname[-1].split('/')[-1])
    dig_in = np.array(dig_in)

    # The tail end of the pulse generates a negative value when passed through diff
    # This method removes the need for a "for" loop

    diff_points = list(np.where(np.diff(dig_in) == diff_val))
    diff_points[1] = diff_points[1]//new_intersample_interval
    change_points = [diff_points[1][diff_points[0]==this_dig_in] \
                    for this_dig_in in range(len(dig_in))]

    # ==============================
    # Write-Out Extracted LFP 
    # ==============================

    # Grab the names of the arrays containing LFP recordings
    lfp_nodes = hf5.list_nodes('/raw_LFP')

    # Make the Parsed_LFP node in the hdf5 file if it doesn't exist, else move on
    if '/Parsed_LFP' in hf5:
        hf5.remove_node('/Parsed_LFP', recursive = True)
    hf5.create_group('/', 'Parsed_LFP')

    # Create array marking which channel were chosen for further analysis
    # Made in root folder for backward compatibility of code
    # Code further below simply enumerates arrays in Parsed_LFP
    if "/Parsed_LFP_channels" in hf5:
            hf5.remove_node('/Parsed_LFP_channels')
    else:
        hf5.create_array('/', 'Parsed_LFP_channels', electrodegroup)
        hf5.flush()

    # Remove dig_ins which are not relevant
    change_points_fin = [change_points[x] for x in range(len(change_points))\
                    if x in dig_in_list]

    # Make markers to slice trials for every dig_on
    all_trial_markers = [[(x-trial_durations[0],x+trial_durations[1]) \
                    for x in this_dig_in_markers] \
                    for this_dig_in_markers in change_points_fin]

    # Extract trials for every channel for every dig_in
    print('Parsing LFPs')
    all_channel_trials = []
    for channel in tqdm(lfp_nodes):
            this_channel_trials = [\
                    np.asarray([channel[marker_tuple[0]:marker_tuple[1]] \
                            for marker_tuple in this_dig_in]) \
                            for this_dig_in in all_trial_markers
                                    ]
            all_channel_trials.append(this_channel_trials)

    # Resort data to have 4 arrays (one for every dig_in) 
    # with dims (channels , trials, time)
    for dig_in in dig_in_list:
            this_taste_LFP = np.asarray([\
                            channel[dig_in] for channel in all_channel_trials])

            # Put the LFP data for this taste in hdf5 file under /Parsed_LFP
            hf5.create_array('/Parsed_LFP', 'dig_in_%i_LFPs' \
                        % (dig_in), this_taste_LFP)
            hf5.flush()
            
    #Delete data
    hf5.remove_node('/raw_LFP', recursive = True)
    hf5.flush()

    # ================================================
    # Make plots to visually check quality of channels 
    # ================================================

    # Code copied from LFP_Spectrogram_Stone.py
    # Might need cleanup

    dig_in_channels = hf5.list_nodes('/digital_in')
    dig_in_LFP_nodes = hf5.list_nodes('/Parsed_LFP')

    #Create dictionary of all parsed LFP arrays
    LFP_data = [np.array(dig_in_LFP_nodes[node][:]) \
            for node in range(len(dig_in_LFP_nodes))]
        
    # =============================================================================
    # #Channel Check
    # =============================================================================
    # Make directory to store the LFP trace plots. 
    # Delete and remake the directory if it exists
    channel_check_dir = os.path.join(dir_name, 'LFP_channel_check')
    if os.path.exists(channel_check_dir):
        shutil.rmtree(channel_check_dir)
    #try:
    #        os.system('rm -r '+'./LFP_channel_check')
    #except:
    #        pass
    os.mkdir(channel_check_dir)
    hdf5_name = os.path.basename(hdf5_path)

    #Check to make sure LFPs are "normal" and allow user to remove any that are not
    ########################################
    # Channel check plots are now made automatically (Abu 2/3/19)
    ########################################
    #if subplot_check is "Yes":
    for taste in range(len(LFP_data)):

            #Set data
            channel_data = np.mean(LFP_data[taste],axis=1).T
            t=np.array(list(range(0,np.size(channel_data,axis=0))))
            
            mean_val = np.mean(channel_data.flatten())
            std_val = np.std(channel_data.flatten())
            #Create figure
            fig,axes = plt.subplots(nrows=np.size(channel_data,axis=1), 
                    ncols=1,sharex=True, sharey=True,figsize=(12, 8), squeeze=False)
            fig.text(0.5, 0.05, 'Milliseconds', ha='center',fontsize=15)
            axes_list = [item for sublist in axes for item in sublist]
            
            for ax, chan in zip(axes.flatten(),range(np.size(channel_data,axis=1))):
            
                    ax = axes_list.pop(0)
                    ax.set_yticks([])
                    ax.plot(np.squeeze(t), np.squeeze(channel_data[:,chan]))
                    ax.set_ylim([mean_val - 3*std_val, mean_val + 3*std_val])
                    h = ax.set_ylabel('Channel %s' %(chan))
                    h.set_rotation(0)
                    ax.vlines(x=trial_durations[0], ymin=np.min(channel_data[:,chan]),
                            ymax=np.max(channel_data[:,chan]), linewidth=4, color='r')
                    
            fig.subplots_adjust(hspace=0,wspace = -0.15)
            fig.suptitle('Dig in {} - '.format(taste) + \
                    '%s - Channel Check: %s' %(taste, 
                    hdf5_name[0:4])+'\n' + 'Raw LFP Traces; Date: %s' \
                                    %(re.findall(r'_(\d{6})', 
                    hdf5_name)[0]),size=16,fontweight='bold')
            fig.savefig(\
                    os.path.join(channel_check_dir,
                    hdf5_name[0:4] + \
                    '_dig_in{}'.format(taste) + \
                    '_ %s_%s' %(re.findall(r'_(\d{6})', hdf5_name)[0],
                        taste) + '_channelcheck.png'))   

    # ==============================
    # Close Out 
    # ==============================
    print("If you want to compress the file to release disk space, " + \
            "run 'blech_hdf5_repack.py' upon completion.")
    hf5.flush()
    hf5.close()

