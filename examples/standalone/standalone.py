from LAMP import Experiment
from pathlib import Path
import matplotlib.pyplot as plt 

# define root folder as this folder
ROOT_FOLDER = Path(__file__).resolve().parents[0]

# define local configurations specific to current machine
local_config = {
    'paths': {
        'data_folder': Path(f"{ROOT_FOLDER}/data/"),
        }
    }

# define global configurations common across analysis
global_config = {
    'setup': {
        'DAQ': 'None'
        },
    'paths': {
         'calibs_folder': './calibs/',
         #'user_diagnostics': 'diagnostics.'
         }
    }

# list our diagnostics
diagnostic_config = {
    'ESpec': {
        'type': 'ESpec',
        'on_startup': False,
        'data_folder': '/ESpec/',
        'data_type': 'image',
        'data_ext': 'tiff',
        #'calib_subfolder': './ESpec/',
        'calib_file': 'ESpec_calib.toml',
        },
}

# create experiment object using the above configurations
ex = Experiment(ROOT_FOLDER, local_config=local_config, global_config=global_config, diagnostic_config=diagnostic_config)

# NOTE: I haven't yet allowed calibrations to be defined in file... not sure best way to how to handle them... 

# load ESpec object and test plot
ESpec = ex.get_diagnostic('ESpec')
ESpec.set_calib('PrimaryCalibration')
#shot_dict = {'date': 20241225, 'run': 'run01', 'shotnum': 1}
shot_filepath = 'SomeTestShot1'
fig, ax = ESpec.plot_proc_shot(shot_filepath, debug=True)
#ax.set_xlim([100,2000])
#ax.set_ylim([-50,50])
plt.tight_layout()

# -----------------------------------------------------------------------------
plt.show(block=True)