"""Interface layer for ALEPH, Colorado in 2022. Uses Michigan DAQ code.
"""
import os
from pathlib import Path
import csv

class DAQ:

    __version = 0.1
    __name = 'ALEPH2022'
    __authors = ['Brendan Kettle']

    def __init__(self, exp_obj):
        print(f'Using {self.__name} DAQ')
        # pass in experiment object
        self.ex = exp_obj
        self.data_folder = self.ex.config['paths']['data_folder']
        return
    
    def _build_shot_path(self, run_folder, diagnostic, shotnum, ext):

        # check file?
        shot_path = f'{self.data_folder}/{run_folder}/{diagnostic}/shot{shotnum}.{ext}'

        return shot_path

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
