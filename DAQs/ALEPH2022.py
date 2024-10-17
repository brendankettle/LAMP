import os
from pathlib import Path
import csv
import re
from ..DAQ import DAQ

class ALEPH2022(DAQ):
    """Interface layer for ALEPH, Colorado in 2022. Uses Michigan DAQ code.
    """

    __version = 0.1
    __name = 'ALEPH2022'
    __authors = ['Brendan Kettle']

    def __init__(self, exp_obj):
        """Initiate parent base DAQ class to get all shared attributes and funcs"""
        super().__init__(exp_obj)
        return
    
    def _build_shot_filepath(self, diagnostic, run_folder, shotnum, ext):
        """This is used internally, and so can be DAQ specific"""
        # check file?
        shot_filepath = f'{self.data_folder}/{run_folder}/{diagnostic}/shot{shotnum}.{ext}'
        return shot_filepath

    # perhaps some of this can move to base class?
    def get_shot_data(self, diag_name, shot_dict):

        diag_config = self.ex.diags[diag_name].config

        # Double check if shot_dict is dictionary; could just be filepath
        if isinstance(shot_dict, dict):
        
            required = ['data_folder','data_ext','data_type']
            for param in required:
                if param not in diag_config:
                    print(f"get_shot_data() error: {self.__name} DAQ requires a config parameter '{param}' for {diag_name}")
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

    def build_time_point(self, shot_dict):
        """Universal function to return a point in time for DAQ, for comparison, say in calibrations
        """
        # Need to break run into date then run number
        # TODO: Burst???
        if 'run' not in shot_dict:
            print('build_time_point() error; at least a run needed')

        print(shot_dict)

        run_str = shot_dict['run']

        # assume date string first
        date_str = str(run_str[0:8])
        year = int(date_str[0:4])
        month = int(date_str[4:6])
        day = int(date_str[6:8])

        # then assume run numbers at end
        # this could fail for refs etc., but should work for most "real" runs?
        m = re.search(r'\d+$', run_str) 
        run = int(m.group())

        # if 'burst' in shot_dict:
        #     burst_str = shot_dict['burst']
        #     m = re.search(r'\d+$', burst_str) # gets last numbers
        #     burst = int(m.group())
        # else:
        #     burst = 0
        burst=0
        if 'shotnum' in shot_dict:
            shotnum = shot_dict['shotnum']
        else:
            shotnum = 0

        # weight the different components to make a unique increasing number?
        time_point = year*1e13 + month*1e11 + day*1e9 + run*1e6 + burst*1000 + shotnum
        return  time_point
