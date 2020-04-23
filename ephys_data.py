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
from visualize import *
from BAKS import BAKS
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm

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
Functions in class can autoload data from specified files according to specified 
paramters
E.g. whether to take single units, 
fast spiking etc (as this data is already in the hdf5 file)
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
        spike_array :: params :: N-D array with time as last dimension
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
            firing_rate[...,bin_inds[0]//fin_step_size] = \
                    np.sum(spike_array[...,bin_inds[0]:bin_inds[1]], axis=-1)

        return firing_rate

    @staticmethod
    def _calc_baks_rate(resolution, dt, spike_array):
        """
        resolution : resolution of output firing rate (sec)
        dt : resolution of input spike trains (sec)
        """
        t = np.arange(0,spike_array.shape[-1]*dt, resolution)
        array_inds = list(np.ndindex((spike_array.shape[:-1])))
        spike_times = [np.where(spike_array[this_inds])[0]*dt \
                for this_inds in array_inds]

        firing_rates = [BAKS(this_spike_times,t) \
                for this_spike_times in tqdm(spike_times)]

        # Put back into array
        firing_rate_array = np.zeros((*spike_array.shape[:-1],len(t)))
        for this_inds, this_firing in zip(array_inds, firing_rates):
            firing_rate_array[this_inds] = this_firing
        return firing_rate_array

    @staticmethod
    def _raster(spike_array):
        inds = np.where(spike_array)
        fig, ax = plt.subplots()
        ax.scatter(inds[1],inds[0])
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
            self.hdf5_name = None

            self.spikes = None
        
        self.firing_rate_params = {
            'type'          :   None,
            'step_size'     :   None,
            'window_size'   :   None,
            'dt'            :   None,
            'baks_resolution' : None,
            'baks_dt'       : None
                }
        
    def extract_and_process(self):
        self.get_unit_descriptors()
        self.get_spikes()
        self.get_firing_rates()
        self.get_lfps()

    def separate_laser_data(self):
        self.separate_laser_spikes()
        self.separate_laser_firing()
        self.separate_laser_lfp()


    def get_hdf5_name(self):
        """
        Look for the hdf5 file in the directory
        """
        if self.hdf5_name is None:
            hdf5_name = glob.glob(
                    os.path.join(self.data_dir, '**.h5'))
            if not len(hdf5_name) > 0:
                raise Exception('No HDF5 file detected')
            elif len(hdf5_name) > 1:
                selection_list = ['{}) {} \n'.format(num,os.path.basename(file)) \
                        for num,file in enumerate(hdf5_name)]
                selection_string = \
                        'Multiple HDF5 files detected, please select a number:\n{}'.\
                                format("".join(selection_list))
                file_selection = input(selection_string)
                self.hdf5_name = hdf5_name[int(file_selection)]
            else:
                self.hdf5_name = hdf5_name[0]

    def get_unit_descriptors(self):
        """
        Extract unit descriptors from HDF5 file
        """
        self.get_hdf5_name() 
        with tables.open_file(self.hdf5_name, 'r+') as hf5_file:
            self.unit_descriptors = hf5_file.root.unit_descriptor[:]

    def check_laser(self):
        with tables.open_file(self.hdf5_name, 'r+') as hf5: 
            dig_in_list = \
                [x for x in hf5.list_nodes('/spike_trains') if 'dig_in' in x.__str__()]
            self.laser_exists = sum([dig_in.__contains__('laser_durations') \
                for dig_in in dig_in_list]) > 0
        
            if self.laser_exists:
                self.laser_durations = [dig_in.laser_durations[:] \
                        for dig_in in dig_in_list]

    def get_spikes(self):
        """
        Extract spike arrays from specified HD5 files
        """
        self.get_hdf5_name() 
        with tables.open_file(self.hdf5_name, 'r+') as hf5: 
            
            dig_in_list = \
                [x for x in hf5.list_nodes('/spike_trains') if 'dig_in' in x.__str__()]
            
            self.spikes = [dig_in.spike_array[:] for dig_in in dig_in_list]

    def separate_laser_spikes(self):
        """
        Separate spike arrays into laser on and off conditions
        """
        if 'laser_exists' not in dir(self):
            self.check_laser()
        if 'spikes' not in dir(self):
            self.get_spikes()
        if self.laser_exists:
            self.on_spikes = np.array([taste[laser>0] for taste,laser in \
                    zip(self.spikes,self.laser_durations)])
            self.off_spikes = np.array([taste[laser==0] for taste,laser in \
                    zip(self.spikes,self.laser_durations)])
        else:
            raise Exception('No laser trials in this experiment')

    def get_lfps(self):
        """
        Extract parsed lfp arrays from specified HD5 files
        """
        self.get_hdf5_name() 
        with tables.open_file(self.hdf5_name, 'r+') as hf5: 

            if 'Parsed_LFP' in hf5.list_nodes('/').__str__():
                lfp_nodes = [node for node in hf5.list_nodes('/Parsed_LFP')\
                        if 'dig_in' in node.__str__()]
                self.lfp_array = np.asarray([node[:] for node in lfp_nodes])
                self.all_lfp_array = \
                        self.lfp_array.\
                            swapaxes(1,2).\
                            reshape(-1, self.lfp_array.shape[1],\
                                    self.lfp_array.shape[-1]).\
                            swapaxes(0,1)
            else:
                raise Exception('Parsed_LFP node absent in HDF5')

    def separate_laser_lfp(self):
        """
        Separate spike arrays into laser on and off conditions
        """
        if 'laser_exists' not in dir(self):
            self.check_laser()
        if 'lfp_array' not in dir(self):
            self.get_lfps()
        if self.laser_exists:
            self.on_lfp = np.array([taste.swapaxes(0,1)[laser>0] for taste,laser in \
                    zip(self.lfp_array,self.laser_durations)])
            self.off_lfp = np.array([taste.swapaxes(0,1)[laser==0] for taste,laser in \
                    zip(self.lfp_array,self.laser_durations)])
            self.all_on_lfp =\
                np.reshape(self.on_lfp,(-1,*self.on_lfp.shape[-2:]))
            self.all_off_lfp =\
                np.reshape(self.off_lfp,(-1,*self.off_lfp.shape[-2:]))
        else:
            raise Exception('No laser trials in this experiment')

    def firing_rate_method_selector(self):
        params = self.firing_rate_params

        def check_firing_rate_params(params, param_name_list):
            param_exists_bool = [True if x in params.keys() else False \
                    for x in param_name_list]
            if not all(param_exists_bool):
                raise Exception('All required firing rate parameters'\
                        ' have not been specified \n{}'.format(\
                        '\n'.join(map(str,\
                        list(zip(param_exists_bool,param_name_list))))))
            param_present_bool = [params[x] is not None \
                    for x in param_name_list]
            if not all(param_present_bool):
                raise Exception('All required firing rate parameters'\
                        ' have not been specified \n{}'.format(\
                        '\n'.join(map(str,\
                        list(zip(param_present_bool,param_name_list))))))

        if params['type'] == 'conv':
            param_name_list = ['step_size','window_size','dt']
            
            # This checks if anything is missing
            # And raises exception if anything missing
            check_firing_rate_params(params,param_name_list)

            # If all good, define the function to be used
            def calc_firing_func(data):
                firing_rate = \
                        self._calc_firing_rates(\
                        step_size = self.firing_rate_params['step_size'],
                        window_size = self.firing_rate_params['window_size'],
                        dt = self.firing_rate_params['dt'],
                        spike_array = data)
                return firing_rate

        if params['type'] == 'baks':
            param_name_list = ['baks_resolution','baks_dt']
            check_firing_rate_params(params,param_name_list)
            def calc_firing_func(data):
                firing_rate = \
                        self._calc_baks_rate(\
                        resolution = self.firing_rate_params['baks_resolution'],
                        dt = self.firing_rate_params['baks_dt'],
                        spike_array = data)
                return firing_rate

        return calc_firing_func

    def get_firing_rates(self):
        """
        Converts spikes to firing rates
        """
        
        if self.spikes is None:
            raise Exception('Run method "get_spikes" first')
        if None in self.firing_rate_params.values():
            raise Exception('Specify "firing_rate_params" first')

        calc_firing_func = self.firing_rate_method_selector()
        self.firing_list = [calc_firing_func(spikes) for spikes in self.spikes]
        #self.firing_list = [self._calc_firing_rates(
        #    step_size = self.firing_rate_params['step_size'],
        #    window_size = self.firing_rate_params['window_size'],
        #    dt = self.firing_rate_params['dt'],
        #    spike_array = spikes)
        #                    for spikes in self.spikes]
        
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

    def separate_laser_firing(self):
        """
        Separate spike arrays into laser on and off conditions
        """
        if 'laser_exists' not in dir(self):
            self.check_laser()
        if 'firing_array' not in dir(self):
            self.get_firing_rates()
        if self.laser_exists:
            self.on_firing = np.array([taste[laser>0] for taste,laser in \
                    zip(self.firing_list,self.laser_durations)])
            self.off_firing = np.array([taste[laser==0] for taste,laser in \
                    zip(self.firing_list,self.laser_durations)])
            self.all_on_firing =\
                np.reshape(self.on_firing,(-1,*self.on_firing.shape[-2:]))
            self.all_off_firing =\
                np.reshape(self.off_firing,(-1,*self.off_firing.shape[-2:]))
        else:
            raise Exception('No laser trials in this experiment')
