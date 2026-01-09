import os
from pathlib import Path
import re
from LAMP.DAQ import DAQ

class Apollo(DAQ):
    """Interface layer for Apollo based DAQs. I.e. ZEUS etc.?
    THIS NEEDS WORK TO BE UPDATED FOR GENERAL BASE CLASS. CURRENTLY PULLED FROM A ZEUS EXPERIMENT FILE
    """

    __version = 0.1
    __name = 'Apollo'
    __authors = ['Brendan Kettle']

    def __init__(self, exp_obj):
        """Initiate parent base DAQ class to get all shared attributes and funcs"""
        super().__init__(exp_obj)
        return
    
    def _build_shot_filepath(self, diag_folder, date, run_folder, shotnum, ext, data_stem=None, burst_folder='Burst0001'):
        """This is used internally, and so can be DAQ specific"""

        if data_stem is not None:
            data_stem = f'{diag_folder}_'
        shot_filepath = Path(f'{self.data_folder}/{date}/{run_folder}/{burst_folder}/{diag_folder}/{data_stem}shot_{str(shotnum).zfill(5)}.{ext}')
        # .tif or .tiff??? this is definitely a bodge...
        if not shot_filepath.is_file():
            if ext.lower() == 'tif':
                ext = 'tiff'
            elif ext.lower() == 'tiff':
                ext = 'tif'
            shot_filepath = Path(f'{self.data_folder}/{date}/{run_folder}/{burst_folder}/{diag_folder}/{data_stem}shot_{str(shotnum).zfill(5)}.{ext}')


        if not shot_filepath.is_file():
            print(f'Error, Could not find {str(shot_filepath)}') 
            return False
        
        return shot_filepath

    def get_shot_data(self, diag_name, shot_dict):

        #
        # TO DO: Move some of this to base class?
        #

        diag_config = self.ex.diags[diag_name].config

        # Double check if shot_dict is dictionary; could just be filepath
        if isinstance(shot_dict, dict):
        
            required = ['data_folder','data_ext','data_type']
            for param in required:
                if param not in diag_config:
                    print(f"get_shot_data() error: {self.__name} DAQ requires a config parameter '{param}' for {diag_name}")
                    return None

            required = ['date','run','shotnum']
            for param in required:
                if param not in shot_dict:
                    print(f"get_shot_data() error: {self.__name} DAQ requires a shot_dict['{param}'] value")
                    return None
                
            # Apollo?
            if 'burst' in shot_dict:
                burst = shot_dict['burst']
            else:
                burst = 'Burst0001'
            if 'data_stem' in diag_config:
                data_stem = diag_config['data_stem']
            else:
                data_stem = None

            shot_filepath = self._build_shot_filepath(diag_config['data_folder'], shot_dict['date'], shot_dict['run'], shot_dict['shotnum'], diag_config['data_ext'], data_stem = data_stem, burst_folder=burst)

            if diag_config['data_type'] == 'image':
                shot_data = self.load_imdata(shot_filepath)
            else:
                shot_data = self.load_data(shot_filepath, file_type=diag_config['data_ext'])

        # raw filepath?
        # TODO: This could be a non-image filte type...
        else:
            # look for file first
            #shot_filepath = os.path.join(Path(self.data_folder), Path(shot_dict.lstrip('\/')))
            shot_filepath = os.path.join(os.path.normpath(self.data_folder.replace('\\', '/')), os.path.normpath(shot_dict.replace('\\', '/')))
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
    
    # def get_runs(self, timeframe):
    #     """List all runs within a given timeframe; all, a day, etc.
    #     """
    #     runs = []

    #     # get all runs?
    #     if timeframe.lower() == 'all':
    #         for run_folder in sorted(os.listdir(self.data_folder)):
    #             if os.path.isdir(Path(self.data_folder + run_folder)):
    #                 runs.append(run_folder)
    #     else:
    #         print('TO DO: Finish other options for get_runs()!')

    #     return runs

    def build_time_point(self, shot_dict):
        """Universal function to return a point in time for DAQ, for comparison, say in calibrations
        """
        # for Zeus, use date / run / shot
        # TODO: Burst???
        date_str = str(shot_dict['date'])
        year = int(date_str[0:4])
        month = int(date_str[4:6])
        day = int(date_str[6:8])
        if 'run' in shot_dict and shot_dict['run']:
            run_str = shot_dict['run']
            m = re.search(r'\d+$', run_str) # gets last numbers
            run = int(m.group())
        else:
            run = 0
        if 'burst' in shot_dict:
            burst_str = shot_dict['burst']
            m = re.search(r'\d+$', burst_str) # gets last numbers
            burst = int(m.group())
        else:
            burst = 0
        if 'shotnum' in shot_dict:
            shotnum = shot_dict['shotnum']
        else:
            shotnum = 0

        # weight the different components to make a unique increasing number?
        time_point = year*1e13 + month*1e11 + day*1e9 + run*1e6 + burst*1000 + shotnum
        return  time_point
    
    def get_shot_dicts(self, diag_name, timeframe, exceptions=None):
        """timeframe can be 'all' or a dictionary containing lists of dates, or runs"""

        diag_config = self.ex.diags[diag_name].config
        data_folder = Path(f"{self.data_folder}/")

        shot_dicts = []

        # scan all folders?
        if isinstance(timeframe, str) and timeframe.lower() == 'all':
            # get date folders
            dates = []
            for dir_name in os.listdir(data_folder):
                if os.path.isdir(os.path.join(data_folder, dir_name)):
                    # add filename to list (quick bodge here to try and catch only real date folders)
                    if len(dir_name) == 8:
                        dates.append(int(dir_name))
        elif isinstance(timeframe, dict) and 'dates' in timeframe:
            dates = timeframe['dates']
        elif isinstance(timeframe, dict) and 'date' in timeframe:
            dates = [timeframe['date']]

        # now that we have dates, for each, get run(s)
        for date in sorted(dates):
            date_folder = os.path.join(data_folder, str(date))
            # runs passed
            if isinstance(timeframe, dict) and 'runs' in timeframe:
                runs = timeframe['runs']
            # single run
            elif isinstance(timeframe, dict) and 'run' in timeframe:
                runs = [timeframe['run']]
            # scan folder
            else:
                runs = []
                for run_name in os.listdir(date_folder):
                    if os.path.isdir(os.path.join(date_folder, run_name)):
                        runs.append(run_name)
            # now we have date and runs
            for run in sorted(runs):
                run_folder = os.path.join(date_folder, str(run))
                
                # bursts passed
                if isinstance(timeframe, dict) and 'bursts' in timeframe:
                    bursts = timeframe['bursts']
                # single burst
                elif isinstance(timeframe, dict) and 'burst' in timeframe:
                    bursts = [timeframe['burst']]
                    
                # scan folder for Burst folders e.g. 25th and later
                else:
                    bursts = []
                    for burst_name in os.listdir(run_folder):
                        if os.path.isdir(os.path.join(run_folder, burst_name)) and burst_name.lower().startswith("burst"):
                            bursts.append(burst_name)
                    
                    # ignore burst if dont exist e.g. per 25th datsa
                    if not bursts:
                        burst = None
    
                if bursts: 
                    # print("Bursts found on this date/run ", bursts)
                    for burst in sorted(bursts):
                        burst_folder = os.path.join(run_folder, str(burst))

                        if 'shotnums' in timeframe:
                            shotnums = timeframe['shotnums']
                        # scanning for shotnums
                        else:
                            shotnums = []
                            run_diag_path = os.path.join(run_folder, burst, str(diag_config['data_folder']).strip('/\\'))
                            for filename in os.listdir(run_diag_path):
                                if os.path.isfile(os.path.join(run_diag_path, filename)) and 'shot' in filename.lower():
                                    if exceptions and filename in exceptions:
                                        print(f'Skipping {filename}')
                                        continue
                                    m = re.search(r'\d+$', os.path.splitext(filename)[0]) # gets last numbers, after extension removed
                                    if m:
                                        shotnums.append(int(m.group()))
                        shotnums = sorted(shotnums)

                        # Append each shotnum with the corresponding burst
                        for shotnum in shotnums:
                            shot_dicts.append({'date': date, 'run': run, 'burst': burst, 'shotnum': shotnum})

    
                else: 
                    print("No bursts found on this date/run")
                    run_diag_path = os.path.join(run_folder, str(diag_config['data_folder']).strip('/\\'))
                    shotnums = []
                    for filename in os.listdir(run_diag_path):
                        if os.path.isfile(os.path.join(run_diag_path, filename)) and 'shot' in filename.lower():
                            if exceptions and filename in exceptions:
                                print(f'Skipping {filename}')
                                continue
                            m = re.search(r'\d+$', os.path.splitext(filename)[0]) # gets last numbers, after extension removed
                            if m:
                                shotnums.append(int(m.group()))
                                
                    shotnums = sorted(shotnums)
                    for shotnum in shotnums:
                        shot_dicts.append({'date': date, 'run': run, 'shotnum': shotnum})

        return shot_dicts
    
    # def get_shot_info(self, run = None, shotnums = None):
    #     """Return shot information from run csv files
    #     """
    #     if run == None:
    #         print(f'Run name required for get_shot_info(run=,) using {self.__name} DAQ')
    #         return None

    #     shot_info_filepath = f'{self.data_folder}/{run}/laserEnergy_{run}.csv'

    #     # DAQ_Shotnum	Timestamp [(hh)(mm)(ss)(centisecond)]	Labview_ShotsTodayNum	Energy_Measurement [J]

    #     # initializing the titles and rows list
    #     headers = []
    #     DAQ_shotnums = []
    #     timestamps = []
    #     labview_shotnums = []
    #     laser_energies = []
        
    #     # reading csv file
    #     if not os.path.isfile(shot_info_filepath):
    #         # no file?
    #         return None
    #     with open(shot_info_filepath, 'r') as csvfile:
    #         # empty?
    #         csv_dict = [row for row in csv.DictReader(csvfile)]
    #         if len(csv_dict) == 0:
    #             return None
    #         csvfile.seek(0)
    #         csvreader = csv.reader(csvfile)
    #         headers = next(csvreader)
    #         for row in csvreader:
    #             DAQ_shotnums.append(int(row[0]))
    #             timestamps.append(row[1])
    #             labview_shotnums.append(int(row[2]))
    #             laser_energies.append(float(row[3]))

    #     shot_info = {'DAQ_shotnums': DAQ_shotnums, 'timestamps': timestamps, 'labview_shotnums': labview_shotnums, 'laser_energies': laser_energies}

    #     # want a specific shot(s)?
    #     if shotnums:
    #         print('TO DO: Allow specific shot selection in _read_shot_info()!')

    #     return shot_info
