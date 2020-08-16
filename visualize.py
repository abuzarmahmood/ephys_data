import numpy as np
import pylab as plt
from scipy.stats import zscore

def _raster(spike_array):
    inds = np.where(spike_array)
    fig, ax = plt.subplots()
    ax.scatter(inds[1],inds[0])
    return ax

def imshow(x, cmap = 'viridis'):
    """
    Decorator function for more viewable firing rate heatmaps
    """
    plt.imshow(x,
            interpolation='nearest',aspect='auto', 
            origin = 'lower', cmap = cmap)

def gen_square_subplots(num, sharex = False, sharey = False):
    """
    number of subplots to generate
    """
    square_len = int(np.ceil(np.sqrt(num)))
    row_num = int(np.ceil(num / square_len))
    fig, ax = plt.subplots(row_num, square_len, 
            sharex = sharex, sharey = sharey)
    return fig, ax

def firing_overview(data, t_vec = None, y_values_vec = None,
                    interpolation = 'nearest',
                    cmap = 'jet',
                    #min_val = None, max_val=None, 
                    cmap_lims = 'individual',
                    subplot_labels = None,
                    zscore_bool = False):
    """
    Takes 3D numpy array as input and rolls over first dimension
    to generate images over last 2 dimensions
    E.g. (neuron x trial x time) will generate heatmaps of firing
        for every neuron
    """

    if zscore_bool:
        data = np.array([zscore(dat,axis=None) for dat in data])

    if cmap_lims == 'shared':
        min_val, max_val = np.repeat(np.min(data,axis=None),data.shape[0]),\
                                np.repeat(np.max(data,axis=None),data.shape[0])
    else:
        min_val,max_val = np.min(data,axis=tuple(list(np.arange(data.ndim)[1:]))),\
                            np.max(data,axis=tuple(list(np.arange(data.ndim)[1:])))
    if t_vec is None:
        t_vec = np.arange(data.shape[-1])
    if y_values_vec is None:
        y_values_vec = np.arange(data.shape[1])

    if data.shape[-1] != len(t_vec):
        raise Exception('Time dimension in data needs to be'\
            'equal to length of time_vec')
    num_nrns = data.shape[0]

    # Plot firing rates
    square_len = np.int(np.ceil(np.sqrt(num_nrns)))
    fig, ax = plt.subplots(square_len,square_len, sharex='all',sharey='all')
    
    nd_idx_objs = []
    for dim in range(ax.ndim):
        this_shape = np.ones(len(ax.shape))
        this_shape[dim] = ax.shape[dim]
        nd_idx_objs.append(
                np.broadcast_to( 
                    np.reshape(
                        np.arange(ax.shape[dim]),
                        this_shape.astype('int')), ax.shape).flatten())
    
    x,y = np.meshgrid(t_vec, y_values_vec)
    if subplot_labels is None:
        subplot_labels = np.zeros(num_nrns)
    if y_values_vec is None:
        y_values_vec = np.arange(data.shape[1])
    for nrn in range(num_nrns):
        plt.sca(ax[nd_idx_objs[0][nrn],nd_idx_objs[1][nrn]])
        plt.gca().set_title('{}:{}'.format(int(subplot_labels[nrn]),nrn))
        plt.gca().pcolormesh(x, y,
                data[nrn],cmap=cmap,
                vmin = min_val[nrn], vmax = max_val[nrn])
    return ax

