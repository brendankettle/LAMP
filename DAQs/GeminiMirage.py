import os
import numpy as np
import re
from pathlib import Path
import sqlite3
import pandas as pd
from LAMP.DAQ import DAQ

class GeminiMirage(DAQ):
    """Interface layer for Gemini, using Mirage
    """

    __version = 0.1
    __name = 'GeminiMirage'
    __authors = ['Brendan Kettle']

    mirage_db = None
    eCat_db = None

    def __init__(self, exp_obj, config=None):
        """Initiate parent base DAQ class to get all shared attributes and funcs"""
        super().__init__(exp_obj, config=config)
        return

    def _build_shot_filepath(self, diagnostic, date, run, shotnum, ext, burst=None):
        """This is used internally, and so can be DAQ specific"""
        if 'shotnum_zfill' in self.config:
            zfill = self.config['shotnum_zfill']
        else:
            zfill = 3 # default?
        if burst:
            burst_str = f'{burst}/'
        else:
            burst_str = ''
        shot_path = Path(f'{self.data_folder}/{diagnostic}/{date}/{run}/{burst_str}Shot{str(shotnum).zfill(zfill)}.{ext}')
        # check file?
        return shot_path
    
    def _read_mirage_db(self, filepath='data.sqlite'):
        """Mirage should generate a data.sqlite file that contains (among other stuff), a table called shot_summary.
        That tables contains columns for: run, shot_or_burst, timestamp, burst_length, gsn.
        Another useful table, shot_acquisitions has flags for all diagnostics, whether they ran (saved?) or not.
        """
        # This doesn't use the DAQ option in the global.toml file yet....

        # load the sqlite into pandas dataframe?
        full_filepath = Path(os.path.join(self.data_folder,filepath))
        if not os.path.exists(Path(full_filepath)):
            print(f'Gemini Mirage Database Error; {full_filepath} not found')

        # TODO: More error handling here!
        con = sqlite3.connect(full_filepath)
        # cur = con.cursor()
        # res = cur.execute('SELECT * FROM shot_summary')
        # all_shot_info = res.fetchall()
        # print(all_shot_info)
        shot_info_df = pd.read_sql_query("SELECT * from shot_summary", con)

        # run, shot_or_burst, timestamp, burst_length, gsn
        # first 8 characters of run is the date. E.g. 20250307/run40

        # save to object for future access (return as well?)
        self.mirage_db = shot_info_df

        con.close()

        return self.mirage_db

    def _read_eCat_db(self, filepath=None):
        """Read an export eCat csv file to a pandas dataframe"""
        if not filepath:
            filepath = self.config['eCat_file']
        full_filepath = Path(os.path.join(self.data_folder,filepath))
        if not os.path.exists(Path(full_filepath)):
            print(f'Gemini eCat Database Error; {full_filepath} not found')
        self.eCat_db = pd.read_csv(full_filepath)
        return self.eCat_db

    def get_shot_info(self, shot_dict, eCat=True):

        if self.mirage_db is None:
            self._read_mirage_db()

        runstr = f"{shot_dict['date']}/{shot_dict['run']}"
        row = self.mirage_db.loc[(self.mirage_db['run'] == runstr) & (self.mirage_db['shot_or_burst'] == str(shot_dict['shotnum']))]
        if row.empty:
            print(f'Error, Could not locate shot info for: {shot_dict}')
            return False
        else:
            gsn = int(row['gsn'].values[0])
            timestamp = row['timestamp'].values[0] # str

            info = {'gsn': gsn, 
                    'timestamp': timestamp}
            
            if eCat and 'eCat_file' in self.config:
                if self.eCat_db is None:
                    self._read_eCat_db()
                row = self.eCat_db.loc[self.eCat_db['Id'] == gsn] # Id is the GSN
                if not row.empty:
                    info.update(dict(zip(row.columns.values, row.values[0])))

            return info

    def build_time_point(self, shot_dict):
        """Universal function to return a point in time for DAQ, for comparison, say in calibrations
        """
        # for Gemini, use date / run / shot
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
        if 'shotnum' in shot_dict:
            shotnum = shot_dict['shotnum']
        else:
            shotnum = 0

        # weight the different components to make a unique increasing number?
        time_point = year*1e10 + month*1e8 + day*1e6 + run*1000 + shotnum
        return  time_point
    
    def file_exists(self, diag_name, shot_dict):
        diag_config = self.ex.diags[diag_name].config
        ext = diag_config['data_ext']
        folder = diag_config['data_folder']
        if 'burst' in shot_dict: 
            burst = shot_dict['burst']
        else:
            burst = None
        if os.path.isfile(self._build_shot_filepath(folder, shot_dict['date'], shot_dict['run'], shot_dict['shotnum'], ext, burst=burst)):
            return True
        else:
            return False

    def get_filepath(self, diag_name, shot_dict):
        """Required function to return shot filepath, given the diagnostic name and a shot dict (or filename str)"""

        diag_config = self.ex.diags[diag_name].config

        # Check if shot_dict is not dictionary; could just be filepath
        if isinstance(shot_dict, str):
            shot_filepath = Path(f'{self.data_folder}/{shot_dict}') # should this be diagnostic folder?
        else:
            required = ['data_folder','data_ext','data_type']
            for param in required:
                if param not in diag_config:
                    print(f"get_shot_data() error: {self.__name} DAQ requires a config parameter '{param}' for {diag_name}")
                    return None
                
            # TO DO: OR can use GSN?
            if 'GSN' in shot_dict:
                shot_dict = self._shot_dict_from_GSN(shot_dict['GSN'])
            
            required = ['date','run','shotnum']
            for param in required:
                if param not in shot_dict:
                    print(f"get_shot_data() error: {self.__name} DAQ requires a shot_dict['{param}'] value")
                    return None
            if 'burst' in shot_dict: 
                burst = shot_dict['burst']
            else:
                burst = None
                
            shot_filepath = self._build_shot_filepath(diag_config['data_folder'], shot_dict['date'], shot_dict['run'], shot_dict['shotnum'], diag_config['data_ext'], burst=burst)

        return shot_filepath
    
    # # perhaps some of this can move to base class?
    # def get_shot_data(self, diag_name, shot_dict):
    #     """Univeral function for returning shot data given a diagnostic and shot dictionary
    #     This probably needs path work to make sure it works on all OS's
    #     """
    #     shot_filepath = self.get_filepath(diag_name, shot_dict)

    #     diag_config = self.ex.diags[diag_name].config
    
    #     if diag_config['data_type'] == 'image':
    #         shot_data = self.load_imdata(shot_filepath)
    #     else:
    #         shot_data = self.load_data(shot_filepath, file_type=diag_config['data_ext'])

    #     return shot_data
    
    def get_shot_dicts(self, diag_name, timeframe, exceptions=None):
        """timeframe can be 'all' or a dictionary containing lists of dates, or runs
        To Do: Still need to add burst support!
        """

        diag_config = self.ex.diags[diag_name].config
        diag_folder = f"{self.data_folder}/{diag_config['data_folder']}"

        shot_dicts = []

        # scan all folders?
        if isinstance(timeframe, str) and timeframe.lower() == 'all':
            # get date folders
            dates = []
            for dir_name in os.listdir(diag_folder):
                if os.path.isdir(os.path.join(diag_folder, dir_name)):
                    # add filename to list (quick bodge here to try and catch only real date folders)
                    if len(dir_name) == 8:
                        dates.append(int(dir_name))
        elif isinstance(timeframe, dict) and 'dates' in timeframe:
            dates = timeframe['dates']
        elif isinstance(timeframe, dict) and 'date' in timeframe:
            dates = [timeframe['date']]

        # now that we have dates, for each, get run(s)
        for date in sorted(dates):
            date_folder = os.path.join(diag_folder, str(date))
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
            # now we have date and runs, get shots
            # TO DO: WHAT ABOUT BURSTS!
            for run in sorted(runs):
                run_folder = os.path.join(date_folder, str(run))
                shotnums = []
                if isinstance(timeframe, dict) and 'shotnum' in timeframe:
                    shotnums = [timeframe['shotnum']]
                elif isinstance(timeframe, dict) and 'shotnums' in timeframe:
                    shotnums = timeframe['shotnums']
                else:
                    if os.path.isdir(run_folder):
                        for filename in os.listdir(run_folder):
                            if os.path.isfile(os.path.join(run_folder, filename)):
                                if 'shot' in filename.lower():
                                    m = re.search(r'\d+$', os.path.splitext(filename)[0]) # gets last numbers, after extension removed
                                    shotnums.append(int(m.group()))
                                    #print(f"{date} / {run} / {shotnums[-1]}")
                        # Convert to a set to get unique numbers, then back to a sorted list
                        shotnums = sorted(list(set(shotnums)))
                    else:
                        print(f'Error; cannot find data folder to scan for shots: {run_folder}')
                        return False

                # OK, build the list to return!
                for shotnum in shotnums:
                    shot_dicts.append({'date': date, 'run': run, 'shotnum': shotnum})

        return shot_dicts

