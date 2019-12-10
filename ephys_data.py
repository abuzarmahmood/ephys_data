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
    def __init__(self, 
            data_dir = None):
        
        """
        data_dirs : where to look for hdf5 file
            : get_data() loads data from this directory
        """
        if data_dir is None:
            data_dir = easygui.diropenbox('Please select directory with HDF5 file')
        else:
            self.data_dir =         data_dir
        
        self.firing_rate_params = {
            'step_size' :   None,
            'window_size' : None,
            'total_time' :  None,
                }
        
    def get_hdf5_name(self):
        #Look for the hdf5 file in the directory
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
        
        # Lists for spikes and trials from different tastes
        self.spikes = []
        self.dig_in_order = []
        self.off_spikes = []
        self.on_spikes = []
        self.all_off_trials = []
        self.all_on_trials = []
        
        # Iterate through tastes and extract spikes from laser on and off conditions
        # If no array for laser durations, put everything in laser off
        
        dig_in_gen = hf5.root.spike_trains._f_iter_nodes()
        for taste in range(len(hf5.root.spike_trains._f_list_nodes())):
            
            this_dig_in = next(dig_in_gen)
            if 'dig_in' in this_dig_in.__str__():
                self.dig_in_order.append(this_dig_in.__str__())
                self.spikes.append(this_dig_in.spike_array[:])
                
                # Swap axes to make it (neurons x trials x time)
                self.spikes[taste] = np.swapaxes(self.spikes[taste], 0, 1)
                
                if this_dig_in.__contains__('laser_durations'):
                    on_trials = np.where(this_dig_in.laser_durations[:] > 0.0)[0]
                    off_trials = np.where(this_dig_in.laser_durations[:] == 0.0)[0]
                
                    self.all_off_trials.append(off_trials + (taste * len(off_trials) * 2))
                    self.all_on_trials.append(on_trials + (taste * len(on_trials) * 2))
                
                    self.off_spikes.append(self.spikes[taste][:, off_trials, :])
                    self.on_spikes.append(self.spikes[taste][:, on_trials, :])
                    
                else:
                    off_trials = np.arange(0,self.spikes[taste].shape[1])
                    self.all_off_trials.append(off_trials + (taste * len(off_trials)))
                    self.off_spikes.append(self.spikes[taste][:, off_trials, :])
                    self.on_spikes.append(None)
                
        if len(self.all_off_trials) > 0: self.all_off_trials = np.concatenate(np.asarray(self.all_off_trials))
        if len(self.all_on_trials) > 0: 
            self.all_on_trials = np.concatenate(np.asarray(self.all_on_trials))
            self.laser_exists = True
        else: 
            self.laser_exists = False
        
        hf5.close()
    
    
    @staticmethod
    def _calc_firing_rates(step_size, window_size, total_time, spike_array):
        """
        spike_array :: params :: 4D array with time as last dimension
        """
        bin_inds = (0,window_size)
        total_bins = int((total_time - window_size + 1) / step_size) + 1
        bin_list = [(bin_inds[0]+step,bin_inds[1]+step) \
                for step in np.arange(total_bins)*step_size ]

        firing_rate = np.empty((spike_array.shape[0],spike_array.shape[1],total_bins))
        for bin_inds in bin_list:
            firing_rate[:,:,bin_inds[0]//step_size] = \
                    np.sum(spike_array[:,:,bin_inds[0]:bin_inds[1]], axis=-1)

        return firing_rate

    def get_firing_rates(self):
        """
        Converts spikes to firing rates
        """
        
        off_spikes = self.off_spikes
        if self.laser_exists:
            on_spikes = self.on_spikes
        off_firing = []
        on_firing = []
        normal_off_firing = []
        normal_on_firing = []
        
        step_size = self.firing_rate_params['step_size']
        window_size = self.firing_rate_params['window_size']
        tot_time = self.firing_rate_params['total_time']
        firing_len = int((tot_time-window_size)/step_size)-1

        self.off_firing = [self._calc_firing_rates(step_size = step_size,
                                            window_size = window_size,
                                            total_time = tot_time,
                                            spike_array = spikes)
                            for spikes in self.off_spikes ]

        
        if self.laser_exists:
            self.on_firing= [self._calc_firing_rates(step_size = step_size,
                                                window_size = window_size,
                                                total_time = tot_time,
                                                spike_array = spikes)
                                for spikes in self.on_spikes]
        
        #(taste x nrn x trial x time)
        if self.laser_exists:
            all_firing_array = np.concatenate(
                    (np.asarray(self.off_firing),np.asarray(self.on_firing)), axis = 2)
        else:
            all_firing_array = np.asarray(self.off_firing)
        self.all_firing_array = all_firing_array
        
        
    def get_normalized_firing(self):
        """
        Converts spikes to firing rates
        """
        # =============================================================================
        # Normalize firing
        # =============================================================================
        all_firing_array = self.all_firing_array
        normal_off_firing = copy.deepcopy(self.off_firing)
        
        # Normalized firing of every neuron over entire dataset
        for m in range(all_firing_array.shape[1]): # nrn
            min_val = np.min(all_firing_array[:,m,:,:]) # Find min and max vals in entire dataset
            max_val = np.max(all_firing_array[:,m,:,:])
            if not (max_val == 0):
                for l in range(len(normal_off_firing)): #taste
                    for n in range(normal_off_firing[0].shape[1]): # trial
                        normal_off_firing[l][m,n,:] = (normal_off_firing[l][m,n,:] - min_val)/(max_val-min_val)
            else:
                for l in range(len(normal_off_firing)): #taste
                    normal_off_firing[l][m,:,:] = 0
                
        self.normal_off_firing = normal_off_firing
        all_off_firing_array = np.asarray(self.normal_off_firing)
        new_shape = (all_off_firing_array.shape[1],
                     all_off_firing_array.shape[2]*all_off_firing_array.shape[0],
                     all_off_firing_array.shape[3])
        
        new_all_off_firing_array = np.empty(new_shape)
        
        for taste in range(all_off_firing_array.shape[0]):
                new_all_off_firing_array[:, taste*all_off_firing_array.shape[2]:(taste+1)*all_off_firing_array.shape[2],:] = all_off_firing_array[taste,:,:,:] 
        
        self.all_normal_off_firing = new_all_off_firing_array
        
        ### ON FIRING ###
        
        # If on_firing exists, then calculate on firing
        if self.laser_exists:
            
            normal_on_firing = copy.deepcopy(self.on_firing)
            
            for m in range(all_firing_array.shape[1]): # nrn
                min_val = np.min(all_firing_array[:,m,:,:])
                max_val = np.max(all_firing_array[:,m,:,:])
                if not (max_val == 0):
                    for l in range(len(normal_on_firing)): #taste
                        for n in range(normal_on_firing[0].shape[1]): # trial
                            normal_on_firing[l][m,n,:] = (normal_on_firing[l][m,n,:] - min_val)/(max_val-min_val)
                else:
                    for l in range(len(normal_on_firing)): #taste
                        normal_on_firing[l][m,:,:] = 0
                    
            self.normal_on_firing = normal_on_firing
            all_on_firing_array = np.asarray(self.normal_on_firing)
            new_all_on_firing_array = np.empty(new_shape)
    
            for taste in range(all_off_firing_array.shape[0]):
                new_all_on_firing_array[:, taste*all_on_firing_array.shape[2]:(taste+1)*all_on_firing_array.shape[2],:] = all_on_firing_array[taste,:,:,:]
                            
            self.all_normal_on_firing = new_all_on_firing_array
        
    def firing_overview(self, 
                        dat_set = None, 
                        zscore_bool = False, 
                        subtract = False, 
                        cmap = 'jet'):

        if dat_set is None:
            if not self.laser_exists:
                dat_set = 'off'
        else:
            raise Exception("Please say 'on' or 'off'")

        # dat_set takes string values: 'on', 'off'
        data = eval('self.all_normal_%s_firing' % dat_set)
        
        # Subtract the average component of firing to
        # (hopefully) leave the taste discriminative components
        if subtract:
            data = data - np.mean(data,axis = 1)[:,np.newaxis,:]
        
        # Zscore the firing of every neuron for visualization
        if zscore_bool:
            data = zscore(data, axis = 1)

        num_nrns = data.shape[0]
        t_vec = np.linspace(start=0, stop = \
                self.firing_rate_params['total_time'], num = data.shape[-1])

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
                            this_shape.astype('int')), 
                        ax.shape).flatten())
        
        for nrn in range(num_nrns):
            plt.sca(ax[nd_idx_objs[0][nrn],nd_idx_objs[1][nrn]])
            plt.gca().set_title(nrn)
            plt.gca().pcolormesh(t_vec, 
                    np.arange(data.shape[1]), 
                    data[nrn,:,:],
                    cmap = cmap)
            #self.imshow(data[nrn,:,:])
        plt.show()
            
    
    def imshow(self,x):
        """
        Decorator function for more viewable firing rate heatmaps
        """
        plt.imshow(x,interpolation='nearest',aspect='auto')
        
