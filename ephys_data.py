import os
import numpy as np
import tables
import copy
import multiprocessing as mp
import pylab as plt
from scipy.special import gamma
from scipy.stats import zscore
import glob
import easygui

#  ______       _                      _____        _         
# |  ____|     | |                    |  __ \      | |        
# | |__   _ __ | |__  _   _ ___ ______| |  | | __ _| |_ __ _  
# |  __| | '_ \| '_ \| | | / __|______| |  | |/ _` | __/ _` | 
# | |____| |_) | | | | |_| \__ \      | |__| | (_| | || (_| | 
# |______| .__/|_| |_|\__, |___/      |_____/ \__,_|\__\__,_| 
#        | |           __/ |                                  
#        |_|          |___/ 

"""
Make a class to streamline data analysis from multiple files
Class has a container for data from different files and functions for analysis
Functions in class can autoload data from specified files according to specified paramters
E.g. whether to take single units, fast spiking etc (as this data is already in the hdf5 file)
"""

class ephys_data():

    ######################
    # Define static methods
    #####################

    @staticmethod
    def _calc_firing_rates(step_size, window_size, dt , spike_array):
        """
        step_size 
        window_size :: params :: In milliseconds. For moving window firing rate
                                calculation
        sampling_rate :: params :: In ms, To calculate total number of bins 
        spike_array :: params :: 3D array with time as last dimension
        """

        if np.sum([step_size % dt , window_size % dt]) > 1e-14:
            raise Exception('Step size and window size must be integer multiples'\
                    ' of the inter-sample interval')

        fin_step_size, fin_window_size = \
            int(step_size/dt), int(window_size/dt)
        total_time = spike_array.shape[-1]

        bin_inds = (0,fin_window_size)
        total_bins = int((total_time - fin_window_size + 1) / fin_step_size) + 1
        bin_list = [(bin_inds[0]+step,bin_inds[1]+step) \
                for step in np.arange(total_bins)*fin_step_size ]

        firing_rate = np.empty((spike_array.shape[0],spike_array.shape[1],total_bins))
        for bin_inds in bin_list:
            firing_rate[:,:,bin_inds[0]//fin_step_size] = \
                    np.sum(spike_array[:,:,bin_inds[0]:bin_inds[1]], axis=-1)

        return firing_rate

    @staticmethod 
    def imshow(x):
        """
        Decorator function for more viewable firing rate heatmaps
        """
        plt.imshow(x,interpolation='nearest',aspect='auto')

    @staticmethod
    def firing_overview(data, time_step = 25, cmap = 'jet'):
        """
        Takes 3D numpy array as input and rolls over first dimension
        to generate images over last 2 dimensions
        E.g. (neuron x trial x time) will generate heatmaps of firing
            for every neuron
        """
        num_nrns = data.shape[0]
        t_vec = np.arange(data.shape[-1])*time_step 

        # Plot firing rates
        square_len = np.int(np.ceil(np.sqrt(num_nrns)))
        fig, ax = plt.subplots(square_len,square_len)
        
        nd_idx_objs = []
        for dim in range(ax.ndim):
            this_shape = np.ones(len(ax.shape))
            this_shape[dim] = ax.shape[dim]
            nd_idx_objs.append(
                    np.broadcast_to( 
                        np.reshape(
                            np.arange(ax.shape[dim]),
                            this_shape.astype('int')), ax.shape).flatten())
        
        for nrn in range(num_nrns):
            plt.sca(ax[nd_idx_objs[0][nrn],nd_idx_objs[1][nrn]])
            plt.gca().set_title(nrn)
            plt.gca().pcolormesh(t_vec, np.arange(data.shape[1]),
                    data[nrn,:,:],cmap=cmap)
        return ax

    ####################
    # Initialize instance
    ###################

    def __init__(self, 
            data_dir = None):
        
        """
        data_dirs : where to look for hdf5 file
            : get_data() loads data from this directory
        """
        if data_dir is None:
            self.data_dir = easygui.diropenbox('Please select directory with HDF5 file')
        else:
            self.data_dir =         data_dir
        
        self.firing_rate_params = {
            'step_size' :   None,
            'window_size' : None,
            'dt' :  None,
                }
        
    def extract_and_process(self):
        self.get_data()
        self.get_firing_rates()

    def get_hdf5_name(self):
        """
        Look for the hdf5 file in the directory
        """
        hdf5_name = glob.glob(
                os.path.join(self.data_dir, '**.h5'))[0]
        if not len(hdf5_name) > 0:
            raise Exception('No HDF5 file detected')
        else:
            self.hdf5_name = hdf5_name


    def get_data(self):
        """
        Extract spike arrays from specified HD5 files
        """
        self.get_hdf5_name() 
        hf5 = tables.open_file(self.hdf5_name, 'r+')
        
        # Iterate through tastes and extract spikes from laser on and off conditions
        # If no array for laser durations, put everything in laser off
        
        dig_in_list = \
            [x for x in hf5.list_nodes('/spike_trains') if 'dig_in' in x.__str__()]
        
        self.spikes = [dig_in.spike_array[:] for dig_in in dig_in_list]
        
        self.laser_exists = sum([dig_in.__contains__('laser_durations') \
                for dig_in in dig_in_list]) > 0
        
        if self.laser_exists:
            self.laser_durations = [dig_in.laser_durations[:] \
                    for dig_in in dig_in_list]

        hf5.close()

    def get_firing_rates(self):
        """
        Converts spikes to firing rates
        """
        
        self.firing_list = [self._calc_firing_rates(
            step_size = self.firing_rate_params['step_size'],
            window_size = self.firing_rate_params['window_size'],
            dt = self.firing_rate_params['dt'],
            spike_array = spikes)
                            for spikes in self.spikes]
        
        if np.sum([self.firing_list[0].shape == x.shape for x in self.firing_list]) ==\
              len(self.firing_list):
            print('All tastes have equal dimensions,' \
                    'concatenating and normalizing')
            
            # Reshape for backward compatiblity
            self.firing_array = np.asarray(self.firing_list).swapaxes(1,2)
            # Concatenate firing across all tastes
            self.all_firing_array = \
                    self.firing_array.\
                        swapaxes(1,2).\
                        reshape(-1, self.firing_array.shape[1],\
                                self.firing_array.shape[-1]).\
                        swapaxes(0,1)
            
            # Calculate normalized firing
            min_vals = [np.min(self.firing_array[:,nrn,:,:],axis=None) \
                    for nrn in range(self.firing_array.shape[1])] 
            max_vals = [np.max(self.firing_array[:,nrn,:,:],axis=None) \
                    for nrn in range(self.firing_array.shape[1])] 
            self.normalized_firing = np.asarray(\
                    [(self.firing_array[:,nrn,:,:] - min_vals[nrn])/\
                        (max_vals[nrn] - min_vals[nrn]) \
                        for nrn in range(self.firing_array.shape[1])]).\
                        swapaxes(0,1)

            # Concatenate normalized firing across all tastes
            self.all_normalized_firing = \
                    self.normalized_firing.\
                        swapaxes(1,2).\
                        reshape(-1, self.normalized_firing.shape[1],\
                                self.normalized_firing.shape[-1]).\
                        swapaxes(0,1)

        else:
            raise Exception('Cannot currently handle different'\
                    'numbers of trials')
