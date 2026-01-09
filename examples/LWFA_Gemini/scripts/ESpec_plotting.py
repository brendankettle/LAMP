from LAMP import Experiment
import matplotlib.pyplot as plt 
import numpy as np

# create experiment object
ex = Experiment('../') # point to root folder

# get ESpec diagnostic
ESpec = ex.get_diagnostic('ESpec')

# define a shot dictionary
shot_dict = {'date': 20250218, 'run': 'run05', 'shotnum': 40}

# plot a processed ESpec shot
fig, ax = ESpec.plot_proc_shot(shot_dict, debug=True)
#ax.set_xlim([100,2000])
##ax.set_ylim([-50,50])
plt.tight_layout()

# ------------------
plt.show(block=True)
