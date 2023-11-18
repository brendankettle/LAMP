"""Interface layer for ALEPH, Colorado in 2022. Uses Michigan DAQ code.
"""
import os
from pathlib import Path
import csv
from .DAQ import DAQ

class ALEPH2022(DAQ):

    __version = 0.1
    __name = 'ALEPH2022'
    __authors = ['Brendan Kettle']

    def __init__(self, exp_obj):
        # pass in experiment object
        self.ex = exp_obj
        self.data_folder = self.ex.config['paths']['data_folder']
        # Initiate parent base DAQ class to get all shared attributes and funcs
        super().__init__()
        return
    
    def _build_shot_filepath(self, diagnostic, run_folder, shotnum, ext):
        # check file?
        shot_filepath = f'{self.data_folder}/{run_folder}/{diagnostic}/shot{shotnum}.{ext}'
        return shot_filepath

    # perhaps some of this can move to base class?
    def get_shot_data(self, diag_name, shot_dict):

        diag_config = self.ex.diags[diag_name].config['setup']

        # Double check if shot_dict is dictionary; could just be filepath
        if isinstance(shot_dict, dict):
        
            required = ['data_folder','data_ext','data_type']
            for param in required:
                if param not in diag_config:
                    print(f"get_shot_data() error: {self.__name} DAQ requires a config['setup'] parameter '{param}' for {diag_name}")
                    return None

            required = ['run','shotnum']
            for param in required:
                if param not in shot_dict:
                    print(f"get_shot_data() error: {self.__name} DAQ requires a shot_dict['{param}'] value")
                    return None

            shot_filepath = self._build_shot_filepath(diag_config['data_folder'], shot_dict['run'], shot_dict['shotnum'], diag_config['data_ext'])

            if diag_config['data_type'] == 'image':
                shot_data = self.load_imdata(shot_filepath)
            else:
                print('Non-image data loading not yet supported... probably need to add text at least?')

        # raw filepath?
        else:
            # look for file first
            shot_filepath = self.data_folder + shot_dict
            if os.path.exists(shot_filepath):
                filepath_no_ext, file_ext = os.path.splitext(shot_filepath)
                img_exts = {".tif",".tiff"}
                # if it's there, try and suss out data type from file extension
                if file_ext in img_exts:
                    shot_data = self.load_imdata(shot_filepath)
                else:
                    print(f"Error; get_shot_data(); could not identify file type for extension: {file_ext}")
            else:
                print(f"Error; get_shot_data(); could not find shot with raw filepath: {shot_filepath}")

        return shot_data
    
    def get_runs(self, timeframe):
        """List all runs within a given timeframe; all, a day, etc.
        """
        runs = []

        # get all runs?
        if timeframe.lower() == 'all':
            for run_folder in sorted(os.listdir(self.data_folder)):
                if os.path.isdir(Path(self.data_folder + run_folder)):
                    runs.append(run_folder)
        else:
            print('TO DO: Finish other options for get_runs()!')

        return runs

    def get_shot_info(self, run = None, shotnums = None):
        """Return shot information from run csv files
        """
        if run == None:
            print(f'Run name required for get_shot_info(run=,) using {self.__name} DAQ')
            return None

        shot_info_filepath = f'{self.data_folder}/{run}/laserEnergy_{run}.csv'

        # DAQ_Shotnum	Timestamp [(hh)(mm)(ss)(centisecond)]	Labview_ShotsTodayNum	Energy_Measurement [J]

        # initializing the titles and rows list
        headers = []
        DAQ_shotnums = []
        timestamps = []
        labview_shotnums = []
        laser_energies = []
        
        # reading csv file
        if not os.path.isfile(shot_info_filepath):
            # no file?
            return None
        with open(shot_info_filepath, 'r') as csvfile:
            # empty?
            csv_dict = [row for row in csv.DictReader(csvfile)]
            if len(csv_dict) == 0:
                return None
            csvfile.seek(0)
            csvreader = csv.reader(csvfile)
            headers = next(csvreader)
            for row in csvreader:
                DAQ_shotnums.append(int(row[0]))
                timestamps.append(row[1])
                labview_shotnums.append(int(row[2]))
                laser_energies.append(float(row[3]))

        shot_info = {'DAQ_shotnums': DAQ_shotnums, 'timestamps': timestamps, 'labview_shotnums': labview_shotnums, 'laser_energies': laser_energies}

        # want a specific shot(s)?
        if shotnums:
            print('TO DO: Allow specific shot selection in _read_shot_info()!')

        return shot_info
