import os
import matplotlib.pyplot as plt

os.chdir('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data

dat = \
ephys_data('/media/bigdata/Abuzar_Data/AM11/AM11_extracted/AM11_4Tastes_191030_114043_copy')

dat.firing_rate_params = dict(zip(('step_size','window_size','dt'),
                                    (25,250,1)))

dat.extract_and_process()
dat.firing_overview(dat.all_normalized_firing);plt.show()
dat.firing_overview(dat.all_firing_array);plt.show()
